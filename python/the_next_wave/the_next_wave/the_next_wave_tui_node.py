#!/usr/bin/env python3

"""
Terminal dashboard for TheNextWave, built on the buoy API Interface.

Overrides `prediction_callback` to receive `/wave_predictions` and renders three
things with rich + plotext, in the terminal, over ssh, with no X server:

  * a fit-quality table, one row per buoy
  * measured vs reconstructed elevation at one buoy
  * the WEC target: recent actual, the model over the window, and the forecast

The table is the point of the thing. `recon/meas` and `rho` say whether the
inversion reproduces the water it was handed, which is upstream of every
forecast number; `skill` repeats what the solver node publishes so in-window fit
and out-of-window forecast sit side by side.

Measurements and reconstruction are only present when the solver node has
`publish_measurements` / `publish_reconstruction` enabled. They are off in the
deployment config because serializing them dominates the publish step, so the
panels say which flag to turn on rather than drawing an empty axis.
"""

from __future__ import annotations

from collections import deque
import os
import select
import sys
import termios
import threading
import time
import traceback
import tty
from typing import Callable, Optional

from buoy_api.interface import Interface
from buoy_interfaces.msg import WavePredictionOutput
import numpy as np
import plotext as plt
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rich.ansi import AnsiDecoder
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def reshape_f(vec, n_samples: int, n_buoys: int) -> np.ndarray:
    """Unflatten a Fortran-ordered (N x B) message array, or give back empty."""
    a = np.asarray(vec, dtype=float)
    if n_samples <= 0 or n_buoys <= 0 or a.size != n_samples * n_buoys:
        return np.zeros((0, 0), dtype=float)
    return a.reshape((n_samples, n_buoys), order='F')


def fit_stats(meas: np.ndarray, recon: np.ndarray) -> dict:
    """
    Compare one buoy's measured and reconstructed elevation.

    `ratio` is rms(recon)/rms(meas): 1.0 means the fit carries the right amount
    of energy, well under 1 means it is barely fitting anything.

    `rho` is the correlation, which is where phase errors show up. The two are
    independent failures -- a fit can have the right energy in the wrong place,
    and `skill` (1 - SSE/SS) folds both together the same way the published
    forecast skill does, so the three columns read as one story.
    """
    out = {'rms_meas': np.nan, 'rms_recon': np.nan,
           'ratio': np.nan, 'rho': np.nan, 'skill': np.nan}
    m = np.asarray(meas, dtype=float).reshape((-1,))
    r = np.asarray(recon, dtype=float).reshape((-1,))
    n = min(m.size, r.size)
    if n < 2:
        return out
    m, r = m[:n], r[:n]
    good = np.isfinite(m) & np.isfinite(r)
    if good.sum() < 2:
        return out
    m, r = m[good], r[good]

    ss_m = float(np.sum(m * m))
    out['rms_meas'] = float(np.sqrt(np.mean(m * m)))
    out['rms_recon'] = float(np.sqrt(np.mean(r * r)))
    if out['rms_meas'] > 0.0:
        out['ratio'] = out['rms_recon'] / out['rms_meas']
    if ss_m > 0.0:
        out['skill'] = 1.0 - float(np.sum((m - r) ** 2)) / ss_m

    sm, sr = float(np.std(m)), float(np.std(r))
    if sm > 0.0 and sr > 0.0:
        out['rho'] = float(np.mean((m - m.mean()) * (r - r.mean())) / (sm * sr))
    return out


def colour_for(value: float, good: float, bad: float) -> str:
    """Pick a rich style by where `value` falls between `bad` and `good`."""
    if not np.isfinite(value):
        return 'dim'
    if good > bad:
        if value >= good:
            return 'bold green'
        return 'yellow' if value > bad else 'bold red'
    if value <= good:
        return 'bold green'
    return 'yellow' if value < bad else 'bold red'


def fmt(value: float, spec: str = '7.3f') -> str:
    return '--' if value is None or not np.isfinite(value) else format(value, spec)


class RawKeys:
    """
    Read single keypresses without waiting for Enter, restoring the terminal.

    cbreak rather than raw mode, so Ctrl-C still raises KeyboardInterrupt. A
    non-tty stdin (piped or redirected) degrades to never reporting a key
    instead of failing.
    """

    def __init__(self) -> None:
        self.fd: Optional[int] = None
        self.saved = None

    def __enter__(self) -> 'RawKeys':
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                self.fd = sys.stdin.fileno()
                self.saved = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
        except Exception:
            self.fd = None
            self.saved = None
        return self

    def __exit__(self, *exc) -> bool:
        if self.fd is not None and self.saved is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            except Exception:
                pass
        return False

    def key(self) -> str:
        """Return a pending keypress, or '' if none is waiting."""
        if self.fd is None:
            return ''
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            return sys.stdin.read(1) if ready else ''
        except Exception:
            return ''


class PlotextView:
    """
    A rich renderable that redraws a plotext figure at the size rich gives it.

    plotext is global and stateful, so the builder gets called inside the render
    pass with the panel's measured width and height and must do its own `clf()`.
    """

    def __init__(self, builder: Callable[[int, int], str]) -> None:
        self.builder = builder
        self.decoder = AnsiDecoder()

    def __rich_console__(self, console, options):
        width = int(options.max_width or console.width)
        height = int(options.height or console.height or 20)
        canvas = self.builder(max(width, 10), max(height, 5))
        yield from self.decoder.decode(canvas)


class TheNextWaveTuiNode(Interface):
    """Subscribes via the buoy API and drives a rich/plotext dashboard."""

    def __init__(self) -> None:
        # No PB microcontroller services are used here, so don't block on them.
        super().__init__('the_next_wave_tui_node', check_for_services=False)

        # Live deployment: wall clock, never /clock. The tick timer below runs
        # on the node clock and would never fire on sim time without a /clock
        # publisher.
        self.use_sim_time(False)

        self.declare_parameter('buoy_index', 0)
        self.declare_parameter('history_sec', 60.0)
        self.declare_parameter('refresh_hz', 4.0)
        self.declare_parameter('plot_theme', 'pro')
        self.declare_parameter('plot_marker', 'braille')
        self.declare_parameter('headless', False)

        self.buoy_index = int(self.get_parameter('buoy_index').value)
        self.history_sec = float(self.get_parameter('history_sec').value)
        self.refresh_hz = float(self.get_parameter('refresh_hz').value)
        self.plot_theme = str(self.get_parameter('plot_theme').value)
        self.plot_marker = str(self.get_parameter('plot_marker').value)
        # Env var as well as the ROS parameter: a parameter is only useful if
        # overrides actually reach this node, which is itself in question.
        self.headless = (
            bool(self.get_parameter('headless').value)
            or os.environ.get('TNW_HEADLESS', '') not in ('', '0', 'false', 'False')
        )
        if not np.isfinite(self.history_sec) or self.history_sec <= 0.0:
            self.history_sec = 60.0
        if not np.isfinite(self.refresh_hz) or self.refresh_hz <= 0.0:
            self.refresh_hz = 4.0

        self.lock = threading.Lock()
        self.latest: Optional[WavePredictionOutput] = None
        self.n_received = 0
        self.last_rx_wall = 0.0
        self.last_error = ''

        # Fallback WEC history for when the message carries a single sample
        # rather than a series (`publish_wec_series: false`).
        self.wec_hist: deque = deque()

        # Past buoy forecasts at the scored lead, (target_time, z per buoy).
        self.fc_hist: deque = deque()
        self.fc_lead_sec = float('nan')

        self.ensure_prediction_subscription()

        # Proof that the executor is actually servicing this node's callbacks.
        # Without it, "no messages" is ambiguous between a dead spin thread and
        # endpoints that never matched.
        # Set from each decoded message so the key handler knows how far the
        # buoy selection may wrap before any message has been plotted.
        self.n_buoys = 0
        self.quit = False
        self.zoom_sec = 0.0
        self.channel = 'z'

        self.n_ticks = 0
        self.tick_timer = self.create_timer(
            1.0, self.tick_callback, callback_group=self.cb_sub,
        )

    def tick_callback(self) -> None:
        self.n_ticks += 1

    def handle_key(self, k: str) -> None:
        """Apply a keypress: switch the plotted buoy, or quit."""
        n_b = max(1, self.n_buoys)
        if k in ('q', 'Q'):
            self.quit = True
        elif k in ('n', 'N', ' ', '\t'):
            self.buoy_index = (self.buoy_index + 1) % n_b
        elif k in ('p', 'P'):
            self.buoy_index = (self.buoy_index - 1) % n_b
        elif k in ('c', 'C'):
            chans = ('z', 'u', 'v', 'all')
            self.channel = chans[(chans.index(self.channel) + 1) % len(chans)]
        elif k in ('z', 'Z'):
            # 0 means the whole window. The short spans exist to put the
            # trailing edge on screen: a fit that comes apart in its last few
            # seconds is invisible across 256 s but is exactly where the
            # forecast starts.
            spans = (0.0, 60.0, 30.0, 15.0)
            self.zoom_sec = spans[(spans.index(self.zoom_sec) + 1) % len(spans)]
        elif k.isdigit() and k != '0':
            i = int(k) - 1
            if i < n_b:
                self.buoy_index = i

    def ensure_prediction_subscription(self) -> None:
        """
        Guarantee a /wave_predictions subscription exists.

        The base Interface only subscribes when it finds a callback name from
        its own internal list in this class's __dict__, so the override only
        takes effect if the name matches that version of buoy_api exactly. Look
        at what it actually created and fill the gap, instead of trusting the
        name to line up.
        """
        self.subs_ = getattr(self, 'subs_', [])
        existing = [s.topic_name for s in self.subs_]
        if any(t.rstrip('/').endswith('wave_predictions') for t in existing):
            self.get_logger().info(f'Interface already subscribed: {existing}')
            return

        self.subs_.append(self.create_subscription(
            WavePredictionOutput,
            '/wave_predictions',
            self.prediction_callback,
            10,
            callback_group=self.cb_sub,
        ))
        self.get_logger().info(
            'Interface did not subscribe to /wave_predictions '
            f'(it made {existing}); subscribed directly.'
        )

    def prediction_callback(self, data: WavePredictionOutput) -> None:
        # A MultiThreadedExecutor runs callbacks on a thread pool and never
        # retrieves the future, so anything raised here vanishes without a
        # traceback and the display simply never updates. Catch it and put it
        # on screen.
        try:
            with self.lock:
                self.latest = data
                self.n_received += 1
                self.last_rx_wall = self.now_wall()
                self.last_error = ''
                self.buffer_forecast(data)
        except Exception:
            with self.lock:
                self.last_error = traceback.format_exc()

    def buffer_forecast(self, msg: WavePredictionOutput) -> None:
        """
        Keep each window's buoy forecast at the scored lead, by target time.

        This is the raw material of forecast_skill: a forecast made at one
        window for an instant that later arrives as a measurement. Buffering it
        here lets the buoy plot draw forecast against measurement on one axis,
        which is the same comparison the skill number sums over. Buffered under
        the caller's lock, once per message.
        """
        n_b = int(msg.n_buoy_prediction_buoys)
        n_lead = int(msg.n_buoy_prediction_leads)
        pts = msg.buoy_predictions
        if n_b <= 0 or n_lead <= 0 or len(pts) < n_b * n_lead:
            return

        # Location-major: buoy 0 at every lead, then buoy 1, ...
        fz = np.array([p.elevation for p in pts[:n_b * n_lead]],
                      dtype=float).reshape((n_b, n_lead))
        ft = np.array([p.time for p in pts[:n_b * n_lead]],
                      dtype=float).reshape((n_b, n_lead))

        lead = float(msg.forecast_skill_lead_sec)
        if not np.isfinite(lead) or lead <= 0.0:
            lead = 1.0
        leads = ft[0, :] - float(msg.window_end_time)
        i = int(np.argmin(np.abs(leads - lead)))
        self.fc_lead_sec = float(leads[i])

        self.fc_hist.append((float(ft[0, i]), fz[:, i].copy()))
        cutoff = float(msg.window_end_time) - self.history_sec
        while self.fc_hist and self.fc_hist[0][0] < cutoff:
            self.fc_hist.popleft()

    def now_wall(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def snapshot(self) -> tuple[Optional[WavePredictionOutput], int, float]:
        with self.lock:
            return self.latest, self.n_received, self.last_rx_wall

    def waiting_panel(self) -> Panel:
        """
        Explain the wait rather than just showing it.

        Nothing arriving has three distinct causes -- no subscription, no
        publisher, or a callback that is throwing -- and under a
        MultiThreadedExecutor all three look identical from outside. Print the
        state that tells them apart.
        """
        topics = [s.topic_name for s in getattr(self, 'subs_', [])]
        try:
            n_pub = self.count_publishers('/wave_predictions')
        except Exception:
            n_pub = -1

        g = Table.grid(padding=(0, 2))
        g.add_column(justify='right', style='dim')
        g.add_column()
        g.add_row('state', Text('waiting for /wave_predictions', style='yellow'))
        g.add_row('subscribed to', ', '.join(topics) or Text('NOTHING', style='bold red'))
        g.add_row(
            'publishers seen',
            Text(str(n_pub), style='bold red' if n_pub == 0 else 'green'),
        )
        g.add_row(
            'executor ticks',
            Text(str(self.n_ticks),
                 style='bold red' if self.n_ticks == 0 else 'green'),
        )
        g.add_row('messages', str(self.n_received))
        body = g
        if self.last_error:
            body = Group(g, Text(self.last_error, style='red'))
        return Panel(body, title='the next wave', border_style='cyan')

    # ---------- decoding ----------

    def decode(self, msg: WavePredictionOutput) -> dict:
        """Pull everything the panels need out of one message."""
        m = msg.measurements
        r = msg.reconstruction
        n_s, n_b = int(m.n_samples), int(m.n_buoys)

        r_s, r_b = int(r.n_samples), int(r.n_buoys)
        d = {
            't_meas': np.asarray(m.time, dtype=float),
            'z_meas': reshape_f(m.z_meas, n_s, n_b),
            'u_meas': reshape_f(m.u_meas, n_s, n_b),
            'v_meas': reshape_f(m.v_meas, n_s, n_b),
            't_rec': np.asarray(r.time, dtype=float),
            'z_recon': reshape_f(r.z_recon, r_s, r_b),
            'u_recon': reshape_f(r.u_recon, r_s, r_b),
            'v_recon': reshape_f(r.v_recon, r_s, r_b),
            'labels': [str(s) for s in msg.buoy_prediction_labels],
            'solve_swifts': [str(s) for s in msg.solve_swifts],
            'window_end': float(msg.window_end_time),
            'window_start': float(msg.window_start_time),
        }

        # Target forecast, and the model evaluated back over the window.
        d['t_pred'] = np.array([float(p.time) for p in msg.predictions], dtype=float)
        d['z_pred'] = np.array([float(p.elevation) for p in msg.predictions], dtype=float)
        d['t_dense'] = (np.asarray(msg.dense_predictions_time, dtype=float)
                        if msg.has_dense_predictions else np.array([], dtype=float))
        d['z_dense'] = (np.asarray(msg.dense_predictions_z, dtype=float)
                        if msg.has_dense_predictions else np.array([], dtype=float))

        if msg.has_wec_actual_series and len(msg.wec_series_time):
            d['t_wec'] = np.asarray(msg.wec_series_time, dtype=float)
            d['z_wec'] = np.asarray(msg.wec_series_z, dtype=float)
        else:
            if msg.has_wec_actual:
                self.wec_hist.append((float(msg.wec_time), float(msg.wec_z)))
                cutoff = float(msg.wec_time) - self.history_sec
                while self.wec_hist and self.wec_hist[0][0] < cutoff:
                    self.wec_hist.popleft()
            if self.wec_hist:
                arr = np.asarray(self.wec_hist, dtype=float)
                d['t_wec'], d['z_wec'] = arr[:, 0], arr[:, 1]
            else:
                d['t_wec'] = np.array([], dtype=float)
                d['z_wec'] = np.array([], dtype=float)
        return d

    def forecast_stats(self, d: dict, b: int) -> dict:
        """
        Recompute forecast skill for one buoy from the published arrays.

        Independent of the solver node's own bookkeeping: each buffered
        forecast is paired with the measurement interpolated at the instant it
        was made for. Agreeing with the published forecast_skill means the
        metric is sound and the forecast is genuinely bad; disagreeing means
        the metric is. Splitting it into ratio and rho also separates a
        mis-scaled forecast from a mis-timed one, which the single skill
        number cannot -- skill = 2*a*rho - a^2 has two roots, so a badly
        inverted forecast and an over-amplified one give the same value.
        """
        empty = fit_stats(np.array([]), np.array([]))
        t_m = np.asarray(d['t_meas'], dtype=float).reshape((-1,))
        z_all = d['z_meas']
        if not self.fc_hist or t_m.size < 2 or z_all.ndim != 2 or b >= z_all.shape[1]:
            return empty

        n = min(t_m.size, z_all.shape[0])
        t_m, z_m = t_m[:n], z_all[:n, b]
        pairs = [(t, z[b]) for t, z in self.fc_hist
                 if b < z.size and t_m[0] <= t <= t_m[-1]]
        if len(pairs) < 2:
            return empty

        ts = np.array([p[0] for p in pairs], dtype=float)
        fc = np.array([p[1] for p in pairs], dtype=float)
        obs = np.interp(ts, t_m, z_m)
        return fit_stats(obs, fc)

    def buoy_names(self, d: dict, n: int) -> list[str]:
        names = list(d['labels'])
        while len(names) < n:
            names.append(f'buoy{len(names)}')
        return names[:n]

    def selected_buoy(self, n_b: int) -> int:
        if n_b <= 0:
            return 0
        return int(np.clip(self.buoy_index, 0, n_b - 1))

    # ---------- panels ----------

    def header_panel(self, msg, d: dict, n_rx: int, age: float) -> Panel:
        g = Table.grid(padding=(0, 2))
        g.add_column(justify='right', style='dim')
        g.add_column()

        span = d['window_end'] - d['window_start']
        skill = float(msg.forecast_skill)
        g.add_row('window', f"{span:7.1f} s   {int(msg.n_measurements)} samples")
        g.add_row('solve', ', '.join(d['solve_swifts']) or '--')
        g.add_row(
            'solver',
            f"{float(msg.solve_time) * 1e3:7.0f} ms   "
            f"err {fmt(float(msg.solver_error), '.3e')}   "
            f"obj {fmt(float(msg.solver_objective), '.3e')}",
        )
        g.add_row(
            'forecast skill',
            Text(
                f"{fmt(skill)}  @ {float(msg.forecast_skill_lead_sec):.1f} s lead"
                f"   (n={int(msg.forecast_skill_n_scored)})",
                style=colour_for(skill, 0.8, 0.0),
            ),
        )
        g.add_row('messages', f"{n_rx}   last {age:.1f} s ago")
        return Panel(g, title='the next wave', border_style='cyan')

    def wavespec_panel(self, msg) -> Panel:
        """
        Bulk wavespec driving this window's solve.

        Lp is derived, not published: the deep-water wavelength at Tp. It is
        here because every spatial phase in the solve is a fraction of it, so
        it is the scale buoy separations have to be read against.
        """
        g = Table.grid(padding=(0, 2))
        g.add_column(justify='right', style='dim')
        g.add_column()

        if not msg.has_wavespec_bulk:
            return Panel(Text('no bulk wavespec in this message', style='dim'),
                         title='wavespec', border_style='cyan')

        hs = float(msg.wavespec_hs)
        tp = float(msg.wavespec_tp)
        freqs = np.asarray(msg.frequencies, dtype=float)
        dirs = np.asarray(msg.directions, dtype=float)

        g.add_row('Hs', f"{fmt(hs, '5.2f')} m")
        g.add_row('Tp', f"{fmt(tp, '5.1f')} s")
        g.add_row(
            'Tm01 / Tm02',
            f"{fmt(float(msg.wavespec_tm01), '5.1f')} / "
            f"{fmt(float(msg.wavespec_tm02), '5.1f')} s",
        )
        g.add_row(
            'Dp / Dm',
            f"{fmt(float(msg.wavespec_dp), '5.1f')} / "
            f"{fmt(float(msg.wavespec_dm), '5.1f')} deg True FROM",
        )
        g.add_row('spread @ peak', f"{fmt(float(msg.wavespec_spreadp), '5.1f')} deg")

        lp = 9.81 * tp * tp / (2.0 * np.pi) if np.isfinite(tp) and tp > 0 else np.nan
        g.add_row('Lp (deep)', f"{fmt(lp, '6.1f')} m")

        if freqs.size:
            g.add_row(
                'grid',
                f"{freqs.size} freq x {dirs.size} dir   "
                f"{freqs.min():.3f}-{freqs.max():.3f} Hz",
            )
        return Panel(g, title='wavespec', border_style='cyan')

    def stats_panel(self, msg, d: dict) -> Panel:
        z_meas, z_recon = d['z_meas'], d['z_recon']
        n_b = max(z_meas.shape[1] if z_meas.ndim == 2 else 0,
                  len(msg.forecast_skill_by_buoy))
        names = self.buoy_names(d, n_b)
        sel = self.selected_buoy(n_b)
        by_buoy = np.asarray(msg.forecast_skill_by_buoy, dtype=float)
        solve_set = set(d['solve_swifts'])

        t = Table(expand=True, header_style='bold')
        t.add_column('')
        t.add_column('buoy')
        t.add_column('role', style='dim')
        t.add_column('rms meas', justify='right')
        t.add_column('rms recon', justify='right')
        t.add_column('recon/meas', justify='right')
        t.add_column('rho', justify='right')
        t.add_column('fit skill', justify='right')
        # Recomputed here from the published arrays, split into amplitude and
        # phase. skill = 2*a*rho - a^2 is two-to-one, so the published scalar
        # alone cannot say whether a bad forecast is inverted or over-scaled.
        t.add_column('f ratio', justify='right')
        t.add_column('f rho', justify='right')
        t.add_column('fcst skill', justify='right')

        have_recon = z_meas.size > 0 and z_recon.size > 0
        for b in range(n_b):
            if have_recon and b < z_meas.shape[1] and b < z_recon.shape[1]:
                s = fit_stats(z_meas[:, b], z_recon[:, b])
            else:
                s = fit_stats(np.array([]), np.array([]))
            f = self.forecast_stats(d, b)
            fs = float(by_buoy[b]) if b < by_buoy.size else np.nan
            if b == int(msg.verification_buoy_index):
                role = 'verify'
            elif names[b] in solve_set:
                role = 'solve'
            else:
                role = '--'
            t.add_row(
                '>' if b == sel else ' ',
                names[b],
                role,
                fmt(s['rms_meas'], '6.3f'),
                fmt(s['rms_recon'], '6.3f'),
                Text(fmt(s['ratio'], '6.3f'), style=colour_for(s['ratio'], 0.9, 0.5)),
                Text(fmt(s['rho'], '6.3f'), style=colour_for(s['rho'], 0.95, 0.7)),
                Text(fmt(s['skill'], '6.3f'), style=colour_for(s['skill'], 0.8, 0.0)),
                Text(fmt(f['ratio'], '6.3f'), style=colour_for(f['ratio'], 0.9, 0.5)),
                Text(fmt(f['rho'], '6.3f'), style=colour_for(f['rho'], 0.9, 0.5)),
                Text(fmt(fs, '6.3f'), style=colour_for(fs, 0.8, 0.0)),
            )

        body = t if n_b else Text('waiting for a prediction message...', style='dim')
        if n_b and not have_recon:
            body = Group(t, Text(
                'recon columns need publish_measurements and publish_reconstruction',
                style='dim yellow',
            ))
        return Panel(body, title='fit quality', border_style='cyan')

    def buoy_plot(self, d: dict, width: int, height: int) -> str:
        """
        Measured vs reconstruction for one buoy, in one channel or all three.

        z, u and v are rows of the same stacked least-squares problem, so a
        solve can satisfy one and not the others. Seeing them together shows
        whether a good heave fit is being bought at the velocity rows' expense.
        """
        plt.clf()
        plt.plot_size(width, height)

        z_meas, z_recon = d['z_meas'], d['z_recon']
        n_b = z_meas.shape[1] if z_meas.ndim == 2 else 0
        if n_b == 0 or z_recon.size == 0:
            plt.theme(self.plot_theme)
            plt.title('measured vs reconstruction -- '
                      'enable publish_measurements / publish_reconstruction')
            return plt.build()

        b = self.selected_buoy(n_b)
        name = self.buoy_names(d, n_b)[b]
        t0 = d['window_end']
        tm = d['t_meas'] - t0
        tr = (d['t_rec'] - t0) if d['t_rec'].size == d['t_meas'].size else tm

        # Zoom keeps only the trailing `zoom_sec`, where the forecast begins.
        def tail(t_rel: np.ndarray, y: np.ndarray) -> tuple[list, list]:
            n = min(t_rel.size, y.shape[0])
            t_rel, y = t_rel[:n], y[:n]
            if self.zoom_sec > 0.0:
                keep = t_rel >= -self.zoom_sec
                t_rel, y = t_rel[keep], y[keep]
            return list(t_rel), list(y)

        # Forecasts exist for elevation only; u/v panels get no overlay.
        fc = [(t, z[b]) for t, z in self.fc_hist
              if b < z.size and (self.zoom_sec <= 0.0 or t - t0 >= -self.zoom_sec)]
        span = 'full window' if self.zoom_sec <= 0.0 else f'last {self.zoom_sec:.0f}s'

        channels = (('z', 'm'), ('u', 'm/s'), ('v', 'm/s')) \
            if self.channel == 'all' else ((self.channel, 'm' if self.channel == 'z'
                                            else 'm/s'),)
        if len(channels) > 1:
            plt.subplots(len(channels), 1)

        for row, (ch, unit) in enumerate(channels):
            if len(channels) > 1:
                plt.subplot(row + 1, 1)
            plt.theme(self.plot_theme)

            meas, recon = d[f'{ch}_meas'], d[f'{ch}_recon']
            if meas.ndim != 2 or b >= meas.shape[1]:
                plt.title(f'{ch}: not published')
                continue

            mx, my = tail(tm, meas[:, b])
            if mx:
                plt.plot(mx, my, label='measured',
                         marker=self.plot_marker, color='cyan')
            if recon.ndim == 2 and b < recon.shape[1]:
                rx, ry = tail(tr, recon[:, b])
                if rx:
                    plt.plot(rx, ry, label='reconstruction',
                             marker=self.plot_marker, color='orange')

            title = f'{name} {ch} [{span}]'
            if ch == 'z' and fc:
                plt.scatter([t - t0 for t, _ in fc], [z for _, z in fc],
                            label=f'forecast +{self.fc_lead_sec:.0f}s',
                            marker='hd', color='green')
                title += f'  +{self.fc_lead_sec:.0f}s forecast'
            elif ch == 'z' and not self.fc_hist:
                title += '  (overlay needs publish_buoy_predictions)'

            plt.title(f'{title}  (t - window_end, s)')
            plt.xlabel('s')
            plt.ylabel(unit)

        return plt.build()

    def wec_plot(self, d: dict, width: int, height: int) -> str:
        plt.clf()
        plt.plot_size(width, height)
        plt.theme(self.plot_theme)

        t0 = d['window_end']
        drew = False
        if d['t_wec'].size:
            plt.plot(list(d['t_wec'] - t0), list(d['z_wec']),
                     label='wec actual', marker=self.plot_marker, color='white')
            drew = True
        if d['t_dense'].size:
            plt.plot(list(d['t_dense'] - t0), list(d['z_dense']),
                     label='model (in window)', marker=self.plot_marker, color='orange')
            drew = True
        if d['t_pred'].size:
            plt.plot(list(d['t_pred'] - t0), list(d['z_pred']),
                     label='forecast', marker=self.plot_marker, color='green')
            drew = True

        if not drew:
            plt.title('wec target -- no prediction samples yet')
            return plt.build()

        # t = 0 is the newest instant every solve buoy had data for: everything
        # left of it is fit, everything right of it is extrapolation.
        plt.vertical_line(0.0, color='red')
        plt.title('wec target: actual / model / forecast  (t - window_end, s)')
        plt.xlabel('s')
        plt.ylabel('m')
        return plt.build()

    # ---------- render ----------

    def render(self) -> Layout:
        msg, n_rx, last_rx = self.snapshot()
        root = Layout()
        if msg is None:
            root.update(self.waiting_panel())
            return root

        d = self.decode(msg)
        age = max(0.0, self.now_wall() - last_rx)

        n_b = max(d['z_meas'].shape[1] if d['z_meas'].ndim == 2 else 0,
                  len(msg.forecast_skill_by_buoy))

        # Table chrome is 4 lines (top border, header, header rule, bottom
        # border) and the Panel adds 2 more, plus one line for the hint shown
        # when the reconstruction columns are unavailable. Undercounting here
        # silently crops the last buoys off the bottom.
        have_recon = bool(d['z_meas'].size and d['z_recon'].size)
        stats_size = n_b + 6 + (0 if have_recon else 1)
        self.n_buoys = n_b

        root.split_column(
            Layout(name='head', size=9),
            Layout(name='stats', size=stats_size),
            Layout(name='buoy', ratio=1),
            Layout(name='wec', ratio=1),
        )
        root['head'].split_row(
            Layout(name='status'),
            Layout(name='spec', size=48),
        )
        root['head']['status'].update(self.header_panel(msg, d, n_rx, age))
        root['head']['spec'].update(self.wavespec_panel(msg))
        root['stats'].update(self.stats_panel(msg, d))
        root['buoy'].update(Panel(
            PlotextView(lambda w, h: self.buoy_plot(d, w, h)),
            border_style='cyan',
            subtitle='n/p buoy   1-9 select   c z|u|v|all   z zoom   q quit',
            subtitle_align='right',
        ))
        root['wec'].update(Panel(
            PlotextView(lambda w, h: self.wec_plot(d, w, h)),
            border_style='cyan',
        ))
        return root


def check_plotext() -> None:
    """
    Fail at startup, not inside a render, if plotext is the wrong one.

    plotext 6 is a rewrite: these module-level functions are gone, plotting
    goes through signal objects, and build() returns a matrix rather than the
    ANSI string the rich panels decode. Without this check the first symptom is
    a traceback several frames deep inside rich's renderer.
    """
    required = ('clf', 'plot_size', 'theme', 'plot', 'scatter',
                'vertical_line', 'build', 'title', 'xlabel', 'ylabel')
    missing = [n for n in required if not hasattr(plt, n)]
    if missing:
        raise SystemExit(
            f'[tui] plotext at {getattr(plt, "__file__", "?")} '
            f'(version {getattr(plt, "__version__", "?")}) is missing '
            f'{missing}. This node needs the 5.x API: '
            f'pip install -U "plotext>=5.3.2,<6"'
        )


def main(args=None):
    # Unconditional, before anything can swallow it: says which file is really
    # running and what it was told, without depending on ROS parameters.
    print(f'[tui] module={__file__} argv={sys.argv} '
          f'TNW_HEADLESS={os.environ.get("TNW_HEADLESS", "")!r}',
          file=sys.stderr, flush=True)
    check_plotext()

    rclpy.init(args=args)
    node = TheNextWaveTuiNode()

    # Interface requires a MultiThreadedExecutor, and its spin() docstring says
    # a non-blocking spin driven from your own loop is the supported
    # alternative. Do that in the main thread: rich.Live needs the main thread
    # only for output, and spinning here means there is no second thread whose
    # liveness has to be reasoned about.
    executor = MultiThreadedExecutor()
    added = executor.add_node(node)

    period = 1.0 / node.refresh_hz

    if node.headless:
        # Same executor, same loop, no rich. If ticks climb here but not in the
        # TUI, the display layer is at fault; if they stay at zero in both, it
        # never was.
        # add_node returns False if the node already belongs to an executor,
        # in which case spin_once here services nothing. timers/subscriptions
        # say whether the entities exist on the node at all.
        # Node.timers / Node.subscriptions are generators on Humble and lists
        # on Jazzy, so count them without assuming either.
        n_timers = sum(1 for _ in node.timers)
        n_subs = sum(1 for _ in node.subscriptions)
        print(f'[tui] headless; add_node={added} '
              f'owned={node.executor is executor} '
              f'timers={n_timers} subs={n_subs} '
              f'topics={[s.topic_name for s in node.subs_]}', flush=True)
        try:
            while rclpy.ok():
                deadline = time.monotonic() + period
                while rclpy.ok() and time.monotonic() < deadline:
                    executor.spin_once(timeout_sec=period / 4.0)
                print(f'[tui] ticks={node.n_ticks} msgs={node.n_received} '
                      f'pubs={node.count_publishers("/wave_predictions")}',
                      flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            executor.remove_node(node)
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return

    console = Console()
    try:
        # auto_refresh=False: no background repaint thread. Every frame is
        # painted synchronously from this loop, so what is on screen is always
        # the frame this iteration built, and there is no second thread
        # contending for the console while the executor runs.
        with RawKeys() as keys, Live(node.render(), console=console,
                                     screen=True, auto_refresh=False) as live:
            while rclpy.ok() and not node.quit:
                # Service callbacks for one frame's worth of time, polling the
                # keyboard between batches so a keypress lands within a frame
                # rather than waiting out the whole budget.
                deadline = time.monotonic() + period
                while rclpy.ok() and time.monotonic() < deadline:
                    executor.spin_once(timeout_sec=period / 4.0)
                    k = keys.key()
                    if k:
                        node.handle_key(k)
                try:
                    frame = node.render()
                except Exception:
                    frame = Panel(Text(traceback.format_exc(), style='red'),
                                  title='render error', border_style='red')
                live.update(frame, refresh=True)
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
