#!/usr/bin/env python3

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import socket
import threading
import time
from typing import Callable

import rclpy

from . import sbgMessageParse
from .readAndDecodeFromEthernetBridge import iter_sbg_headers
from .rolling_csv_logger import RollingCsvLogger


def utc_message_to_epoch_us(data_struct: dict) -> float:
    minute_start = datetime(
        year=data_struct.get('year'),
        month=data_struct.get('month'),
        day=data_struct.get('day'),
        hour=data_struct.get('hour'),
        minute=data_struct.get('min'),
        second=0,
        tzinfo=timezone.utc,
    )

    return (
        minute_start.timestamp() * 1e6
        + data_struct.get('sec') * 1e6
        + data_struct.get('nanosec') * 1e-3
    )


class SbgBridgeService:
    def __init__(
        self,
        *,
        bind: str,
        socket_timeout_sec: float,
        swift_warm_start_us: int,
        port_by_swift: dict[int, int],
        logger,
        data_lock: threading.Lock,
        ingest_swift_sample_locked: Callable[..., None],
    ) -> None:
        self.bind = str(bind)
        self.socket_timeout_sec = float(socket_timeout_sec)
        self.warm_start_us = swift_warm_start_us
        self.port_by_swift = dict(port_by_swift)
        self.logger = logger
        self.data_lock = data_lock
        self.ingest_swift_sample_locked = ingest_swift_sample_locked

        self.stop_event = threading.Event()
        # Live connections, so stop() can shut them down to unblock the readers.
        self.active_conn_by_swift: dict[int, socket.socket] = {}
        self.active_conn_lock = threading.Lock()
        self.threads: list[threading.Thread] = []
        self.partial_by_swift: dict[int, dict] = {}
        self.last_status_t_us_by_swift: dict[int, int] = {}
        self.burst_start_t_us_by_swift: dict[int, int] = {}
        self.last_warn_walltime_by_swift: dict[int, float] = {}
        self.swift_data_logger = {
                22: None,
                23: None,
                24: None,
                25: None,
            }

    def start(self, swift_nums: list[int]) -> None:
        for swift_num in swift_nums:
            port = int(self.port_by_swift[int(swift_num)])
            thread = threading.Thread(
                target=self.server_thread,
                args=(int(swift_num), self.bind, int(port)),
                daemon=True,
            )
            self.threads.append(thread)
            thread.start()
            self.logger.info(f'SBG bridge starting: swift{swift_num} bind {self.bind}:{port}')

    def stop(self) -> None:
        self.stop_event.set()
        # Reader threads block in read() with no timeout, so setting the event
        # alone will not wake them. Shutting the socket down makes the pending
        # read return b'' immediately and the loop exits.
        with self.active_conn_lock:
            conns = list(self.active_conn_by_swift.values())
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def roll_swift_data_loggers(self, swift_num: int) -> None:
        if self.swift_data_logger[swift_num] is None:
            return

        for message_name, data_logger in self.swift_data_logger[swift_num].items():
            try:
                data_logger.roll()
            except Exception as err:
                self.logger.error(
                    f'swift{swift_num} failed to roll '
                    f'{message_name} CSV log: {err}'
                )

    def server_thread(self, swift_num: int, bind: str, port: int) -> None:
        server_sock = None
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server_sock.bind((bind, port))
            except OSError as e:
                if getattr(e, 'errno', None) == 98:
                    self.logger.error(
                        f'swift{swift_num} SBG bridge bind failed on {bind}:{port} '
                        '(address in use). '
                        f'Stop the other process or change swifts.swift{swift_num}.'
                    )
                    return
                raise
            # Backlog > 1 so a SWIFT reconnecting while the previous connection
            # is still being torn down gets queued rather than having its SYN
            # dropped (which looks like a buoy that just never comes back).
            server_sock.listen(8)
            server_sock.settimeout(self.socket_timeout_sec)

            self.logger.info(f'swift{swift_num} SBG bridge listening on {bind}:{port}')

            while rclpy.ok() and not self.stop_event.is_set():
                try:
                    conn, client_addr = server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                reader = None
                try:
                    self.logger.info(f'swift{swift_num} SBG bridge connection from {client_addr}')
                    # Blocking: no read timeout. Keepalive detects a vanished
                    # peer, and stop() shuts the socket down to unblock us.
                    # A timeout here could fire mid-message and leave the
                    # BufferedReader's buffer inconsistent -- likely in the
                    # water, where links drop mid-message.
                    conn.settimeout(None)
                    self.enable_keepalive(swift_num, conn)
                    with self.active_conn_lock:
                        self.active_conn_by_swift[swift_num] = conn
                    reader = conn.makefile('rb')
                    self.connection_loop(swift_num, reader)
                except Exception:
                    self.logger.warn(f'swift{swift_num} SBG bridge connection ended')
                else:
                    self.logger.warn(
                        f'swift{swift_num} SBG bridge connection closed; listening again'
                    )
                finally:
                    with self.active_conn_lock:
                        self.active_conn_by_swift.pop(swift_num, None)
                    for closeable in (reader, conn):
                        try:
                            if closeable is not None:
                                closeable.close()
                        except Exception:
                            pass

        except Exception:
            self.logger.error(f'swift{swift_num} SBG bridge server failed on {bind}:{port}')
        finally:
            if server_sock is not None:
                try:
                    server_sock.close()
                except Exception:
                    pass

    def enable_keepalive(self, swift_num: int, conn: socket.socket) -> None:
        """
        Turn on TCP keepalive so the kernel notices a peer that vanished.

        This is the only mechanism that distinguishes "SWIFT alive but between
        bursts" from "SWIFT gone": recv on a half-open socket times out forever
        and never errors, so silence alone proves nothing. Keepalive probes make
        the kernel tear the socket down, surfacing as a real exception.

        Reports what actually applied. The tuning options are Linux-only, and
        without them the system defaults apply -- typically 7200 s idle plus
        9 x 75 s probes, i.e. over two hours before a dead peer is noticed.
        That is far too slow to be useful here, so a partial application is
        worth warning about rather than silently accepting.
        """
        try:
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError as exc:
            self.logger.warn(
                f'swift{swift_num} could not enable SO_KEEPALIVE ({exc}); '
                'a vanished peer will not be detected'
            )
            return

        # Idle seconds before the first probe, probe interval, failed probes
        # before the connection is declared dead: 15 + 3 x 5 = ~30 s.
        wanted = (('TCP_KEEPIDLE', 15), ('TCP_KEEPINTVL', 5), ('TCP_KEEPCNT', 3))
        missing = []
        for opt_name, value in wanted:
            opt = getattr(socket, opt_name, None)
            if opt is None:
                missing.append(opt_name)
                continue
            try:
                conn.setsockopt(socket.IPPROTO_TCP, opt, value)
            except OSError:
                missing.append(opt_name)

        if missing:
            self.logger.warn(
                f'swift{swift_num} keepalive tuning unavailable ({", ".join(missing)}); '
                'falling back to system defaults, which can take hours to detect '
                'a dead peer, during which the reader stays blocked'
            )

    def connection_loop(self, swift_num: int, reader) -> None:
        for msg_id, msg_class in iter_sbg_headers(reader, stop_event=self.stop_event):
            if not (rclpy.ok() and not self.stop_event.is_set()):
                break

            try:
                data_struct = sbgMessageParse.parseSbgMessage(
                    msg_class,
                    msg_id,
                    connection=reader,
                    printFlag=False,
                )
            except Exception:
                now = time.monotonic()
                last = float(self.last_warn_walltime_by_swift.get(swift_num, 0.0))
                if now - last > 5.0:
                    self.last_warn_walltime_by_swift[swift_num] = now
                    self.logger.warn(f'swift{swift_num} SBG bridge parse error (continuing)')
                continue

            if data_struct is None:
                continue

            self.handle_message(swift_num, msg_id, data_struct)

    def handle_message(self, swift_num: int, msg_id: bytes, data_struct: dict) -> None:
        # print('handler got message:', swift_num, msg_id, data_struct)
        id2name = {
            b'\x01': 'Status',
            b'\x02': 'UtcTime',
            b'\x03': 'ImuData',
            b'\x04': 'Mag',
            b'\x06': 'EkfEuler',
            b'\x07': 'EkfQuat',
            b'\x08': 'EkfNav',
            b'\x09': 'ShipMotion',
            b'\x0d': 'GpsVel',
            b'\x0e': 'GpsPos',
        }

        if msg_id not in id2name:
            return

        message_name = id2name[msg_id]

        try:
            t_us = int(data_struct.get('time_stamp'))

            if message_name == 'Status':
                last_status_t_us = self.last_status_t_us_by_swift.get(swift_num)

                if last_status_t_us is None:
                    self.burst_start_t_us_by_swift[swift_num] = t_us
                elif t_us < last_status_t_us:
                    self.roll_swift_data_loggers(swift_num)
                    self.burst_start_t_us_by_swift[swift_num] = t_us

                self.last_status_t_us_by_swift[swift_num] = t_us

            now = time.time()
            log_data = deepcopy(data_struct)
            log_data.update({
                'handle_time': now,
                'swift_id': swift_num,
                'msg_id': ''.join(f'\\x{byte:02x}' for byte in msg_id),
                'msg_name': message_name,
            })
            first_fields = [
                    'handle_time',
                    'time_stamp',
                    'swift_id',
                    'msg_id',
                    'msg_name',
                ]
            if self.swift_data_logger[swift_num] is None:
                self.swift_data_logger[swift_num] = dict()
            if id2name[msg_id] not in self.swift_data_logger[swift_num]:
                self.swift_data_logger[swift_num].update(
                    {
                        id2name[msg_id]: RollingCsvLogger(
                            f'/mnt/nvme/data/swifts/parsed/swift{swift_num}/{id2name[msg_id]}_parsed.csv',
                            fieldnames=first_fields + sorted(list(set(log_data.keys()) - set(first_fields))),
                        )
                    }
                )
            self.swift_data_logger[swift_num][id2name[msg_id]].write(log_data)
        except Exception as err:
            print('exception in bridge handle_message:', err)
            return

        with self.data_lock:
            if message_name == 'Status':
                rec = {'t_us': t_us}
                self.partial_by_swift[swift_num] = rec
            else:
                rec = self.partial_by_swift.get(swift_num)
                if rec is None:
                    return

            if id2name[msg_id] == 'UtcTime':
                try:
                    rec['t_utc'] = utc_message_to_epoch_us(data_struct)
                except Exception as err:
                    print('handle message error:', err)
            elif id2name[msg_id] == 'ShipMotion':
                try:
                    rec['z'] = float(data_struct.get('heave'))
                except Exception as err:
                    print('handle message error:', err)
            elif id2name[msg_id] == 'GpsVel':
                try:
                    rec['u'] = float(data_struct.get('vel_e'))
                    rec['v'] = float(data_struct.get('vel_n'))
                except Exception as err:
                    print('handle message error:', err)
            elif id2name[msg_id] == 'GpsPos':
                try:
                    rec['lat'] = float(data_struct.get('lat'))
                    rec['lon'] = float(data_struct.get('long'))
                except Exception as err:
                    print('handle message error:', err)

                burst_start_t_us = self.burst_start_t_us_by_swift.get(
                    swift_num,
                    rec['t_us'],
                )

                if (
                    rec['t_us'] - burst_start_t_us >= self.warm_start_us  # 45_000_000
                    and all(k in rec for k in ('z', 'u', 'v', 'lat', 'lon', 't_utc'))
                ):
                    self.ingest_swift_sample_locked(
                        swift_num=swift_num,
                        t_us=float(rec['t_utc']),
                        z=float(rec['z']),
                        u=float(rec['u']),
                        v=float(rec['v']),
                        lat=float(rec['lat']),
                        lon=float(rec['lon']),
                    )

                try:
                    del self.partial_by_swift[swift_num]
                except Exception as err:
                    print('handle message error:', err)
                    pass
