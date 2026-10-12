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
import sys
import threading
import time
import traceback
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

        self.ensure_prediction_subscription()

        # Proof that the executor is actually servicing this node's callbacks.
        # Without it, "no messages" is ambiguous between a dead spin thread and
        # endpoints that never matched.
        self.n_ticks = 0
        self.tick_timer = self.create_timer(
            1.0, self.tick_callback, callback_group=self.cb_sub,
        )

    def tick_callback(self) -> None:
        self.n_ticks += 1

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
        except Exception:
            with self.lock:
                self.last_error = traceback.format_exc()

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

        d = {
            't_meas': np.asarray(m.time, dtype=float),
            'z_meas': reshape_f(m.z_meas, n_s, n_b),
            't_rec': np.asarray(r.time, dtype=float),
            'z_recon': reshape_f(r.z_recon, int(r.n_samples), int(r.n_buoys)),
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
        g.add_row(
            'Hs / Tp / Dp',
            f"{fmt(float(msg.wavespec_hs), '5.2f')} m   "
            f"{fmt(float(msg.wavespec_tp), '5.1f')} s   "
            f"{fmt(float(msg.wavespec_dp), '5.1f')} deg",
        )
        g.add_row('messages', f"{n_rx}   last {age:.1f} s ago")
        return Panel(g, title='the next wave', border_style='cyan')

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
        t.add_column('fcst skill', justify='right')

        have_recon = z_meas.size > 0 and z_recon.size > 0
        for b in range(n_b):
            if have_recon and b < z_meas.shape[1] and b < z_recon.shape[1]:
                s = fit_stats(z_meas[:, b], z_recon[:, b])
            else:
                s = fit_stats(np.array([]), np.array([]))
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
        plt.clf()
        plt.plot_size(width, height)
        plt.theme(self.plot_theme)

        z_meas, z_recon = d['z_meas'], d['z_recon']
        n_b = z_meas.shape[1] if z_meas.ndim == 2 else 0
        if n_b == 0 or z_recon.size == 0:
            plt.title('measured vs reconstruction -- '
                      'enable publish_measurements / publish_reconstruction')
            return plt.build()

        b = self.selected_buoy(n_b)
        name = self.buoy_names(d, n_b)[b]
        t0 = d['window_end']
        tm = d['t_meas'] - t0
        tr = (d['t_rec'] - t0) if d['t_rec'].size == d['t_meas'].size else tm

        n = min(tm.size, z_meas.shape[0])
        plt.plot(list(tm[:n]), list(z_meas[:n, b]),
                 label='measured', marker=self.plot_marker, color='cyan')
        if b < z_recon.shape[1]:
            nr = min(tr.size, z_recon.shape[0])
            plt.plot(list(tr[:nr]), list(z_recon[:nr, b]),
                     label='reconstruction', marker=self.plot_marker, color='orange')

        plt.title(f'{name}: measured vs reconstruction  (t - window_end, s)')
        plt.xlabel('s')
        plt.ylabel('m')
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
        root.split_column(
            Layout(name='head', size=9),
            Layout(name='stats', size=n_b + 4),
            Layout(name='buoy', ratio=1),
            Layout(name='wec', ratio=1),
        )
        root['head'].update(self.header_panel(msg, d, n_rx, age))
        root['stats'].update(self.stats_panel(msg, d))
        root['buoy'].update(Panel(
            PlotextView(lambda w, h: self.buoy_plot(d, w, h)),
            border_style='cyan',
        ))
        root['wec'].update(Panel(
            PlotextView(lambda w, h: self.wec_plot(d, w, h)),
            border_style='cyan',
        ))
        return root


def main(args=None):
    # Unconditional, before anything can swallow it: says which file is really
    # running and what it was told, without depending on ROS parameters.
    print(f'[tui] module={__file__} argv={sys.argv} '
          f'TNW_HEADLESS={os.environ.get("TNW_HEADLESS", "")!r}',
          file=sys.stderr, flush=True)

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
        with Live(node.render(), console=console, screen=True,
                  refresh_per_second=node.refresh_hz) as live:
            while rclpy.ok():
                # Service callbacks for one frame's worth of time, then redraw.
                deadline = time.monotonic() + period
                while rclpy.ok() and time.monotonic() < deadline:
                    executor.spin_once(timeout_sec=period / 4.0)
                try:
                    frame = node.render()
                except Exception:
                    frame = Panel(Text(traceback.format_exc(), style='red'),
                                  title='render error', border_style='red')
                live.update(frame)
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
