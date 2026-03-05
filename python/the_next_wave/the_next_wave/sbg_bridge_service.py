#!/usr/bin/env python3

from collections import OrderedDict
import socket
import threading
import time
from typing import Callable

import rclpy

from . import sbgMessageParse
from .readAndDecodeFromEthernetBridge import iter_sbg_headers


class SbgBridgeService:
    def __init__(
        self,
        *,
        bind: str,
        socket_timeout_sec: float,
        port_by_swift: dict[int, int],
        logger,
        data_lock: threading.Lock,
        ingest_swift_sample_locked: Callable[..., None],
    ) -> None:
        self.bind = str(bind)
        self.socket_timeout_sec = float(socket_timeout_sec)
        self.port_by_swift = dict(port_by_swift)
        self.logger = logger
        self.data_lock = data_lock
        self.ingest_swift_sample_locked = ingest_swift_sample_locked

        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.partial_by_swift: dict[int, 'OrderedDict[int, dict]'] = {}
        self.last_warn_walltime_by_swift: dict[int, float] = {}

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
            server_sock.listen(1)
            server_sock.settimeout(self.socket_timeout_sec)

            self.logger.info(f'swift{swift_num} SBG bridge listening on {bind}:{port}')

            while rclpy.ok() and not self.stop_event.is_set():
                try:
                    conn, client_addr = server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                try:
                    self.logger.info(f'swift{swift_num} SBG bridge connection from {client_addr}')
                    conn.settimeout(self.socket_timeout_sec)
                    self.connection_loop(swift_num, conn)
                except Exception:
                    self.logger.warn(f'swift{swift_num} SBG bridge connection ended')
                finally:
                    try:
                        conn.close()
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

    def connection_loop(self, swift_num: int, conn: socket.socket) -> None:
        for msg_id, msg_class in iter_sbg_headers(conn, stop_event=self.stop_event):
            if not (rclpy.ok() and not self.stop_event.is_set()):
                break

            try:
                data_struct = sbgMessageParse.parseSbgMessage(
                    msg_class,
                    msg_id,
                    connection=conn,
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
        try:
            t_us = int(data_struct.get('time_stamp'))
        except Exception:
            return

        with self.data_lock:
            partial = self.partial_by_swift.setdefault(swift_num, OrderedDict())
            rec = partial.get(t_us)
            if rec is None:
                rec = {'t_us': t_us}
                partial[t_us] = rec

            if msg_id == b'\x09':
                try:
                    rec['z'] = float(data_struct.get('heave'))
                except Exception:
                    pass
            elif msg_id == b'\x0d':
                try:
                    rec['u'] = float(data_struct.get('vel_e'))
                    rec['v'] = float(data_struct.get('vel_n'))
                except Exception:
                    pass
            elif msg_id == b'\x0e':
                try:
                    rec['lat'] = float(data_struct.get('lat'))
                    rec['lon'] = float(data_struct.get('long'))
                except Exception:
                    pass

            if all(k in rec for k in ('z', 'u', 'v', 'lat', 'lon')):
                self.ingest_swift_sample_locked(
                    swift_num=swift_num,
                    t_us=float(rec['t_us']),
                    z=float(rec['z']),
                    u=float(rec['u']),
                    v=float(rec['v']),
                    lat=float(rec['lat']),
                    lon=float(rec['lon']),
                )
                try:
                    del partial[t_us]
                except Exception:
                    pass

            while len(partial) > 100:
                try:
                    partial.popitem(last=False)
                except Exception:
                    break
