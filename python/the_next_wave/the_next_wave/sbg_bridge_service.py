#!/usr/bin/env python3

from collections import OrderedDict
from copy import deepcopy
import socket
import threading
import time
from typing import Callable

import rclpy

from . import sbgMessageParse
from .readAndDecodeFromEthernetBridge import iter_sbg_headers
from .rolling_csv_logger import RollingCsvLogger


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
        self.swift_data_logger = {
                22: None,  # RollingCsvLogger('/mnt/nvme/data/swifts/parsed/swift22_parsed.csv')
                23: None,  # RollingCsvLogger('/mnt/nvme/data/swifts/parsed/swift23_parsed.csv')
                24: None,  # RollingCsvLogger('/mnt/nvme/data/swifts/parsed/swift24_parsed.csv')
                25: None,  # RollingCsvLogger('/mnt/nvme/data/swifts/parsed/swift25_parsed.csv')
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

        message_name = id2name.get(msg_id)
        if message_name not in ('ShipMotion', 'GpsVel', 'GpsPos'):
            return

        try:
            t_us = int(data_struct.get('time_stamp'))
            #if msg_id in [b'\x09', b'\x0d', b'\x0e']:
            #    print(f'[{time.monotonic()}] handler got {id2name[msg_id]}: message time = {t_us}')
            #else:
            #    print('handler got unknown msg_id:', msg_id)

            now = time.monotonic()
            log_data = deepcopy(data_struct)
            log_data.update({'handle_time': now, 'swift_id': swift_num, 'msg_id': f'\\x{msg_id[0]:02x}, 'msg_name': id2name[msg_id]})
            first_fields = [
                    'time_stamp',
                    'handle_time',
                    'swift_id',
                    'msg_id',
                    'msg_name',
                ]
            if self.swift_data_logger[swift_num] is None:
                self.swift_data_logger[swift_num] = dict()
            elif id2name[msg_id] not in self.swift_data_logger[swift_num]:
                self.swift_data_logger[swift_num].update({id2name[msg_id]: RollingCsvLogger(
                    f'/mnt/nvme/data/swifts/parsed/swift{swift_num}/{id2name[msg_id]}_parsed.csv',
                    fieldnames=first_fields + sorted(list(set(log_data.keys()) - set(first_fields))),
                    when='H',
                    interval=1,
                    utc=True,
                    max_bytes=0,
                )})
            self.swift_data_logger[swift_num][id2name[msg_id]].write(log_data)
        except Exception as err:
            print('exception in bridge handle_message:', err)
            return

        with self.data_lock:
            partial = self.partial_by_swift.setdefault(swift_num, OrderedDict())
            message_fields = {
                'ShipMotion': ('z',),
                'GpsVel': ('u', 'v'),
                'GpsPos': ('lat', 'lon'),
            }[message_name]
            rec_key = None
            rec = None
            best_distance = 100_001
            for candidate_key, candidate_rec in partial.items():
                if any(field in candidate_rec for field in message_fields):
                    continue
                distance = abs(t_us - candidate_key)
                if distance <= 100_000 and distance < best_distance:
                    rec_key = candidate_key
                    rec = candidate_rec
                    best_distance = distance

            if rec is None:
                rec_key = t_us
                rec = {'t_us': t_us}
                partial[rec_key] = rec

            if message_name == 'ShipMotion':
                try:
                    rec['z'] = float(data_struct.get('heave'))
                    rec['ship_motion_t_us'] = t_us
                except Exception as err:
                    print('handle message error:', err)
            elif message_name == 'GpsVel':
                try:
                    rec['u'] = float(data_struct.get('vel_e'))
                    rec['v'] = float(data_struct.get('vel_n'))
                    rec['gps_vel_t_us'] = t_us
                except Exception as err:
                    print('handle message error:', err)
            elif message_name == 'GpsPos':
                try:
                    rec['lat'] = float(data_struct.get('lat'))
                    rec['lon'] = float(data_struct.get('long'))
                    rec['gps_pos_t_us'] = t_us
                except Exception as err:
                    print('handle message error:', err)

            if all(k in rec for k in ('z', 'u', 'v', 'lat', 'lon')):
                self.ingest_swift_sample_locked(
                    swift_num=swift_num,
                    t_us=float((rec['gps_vel_t_us'] + rec['gps_pos_t_us']) / 2.0),
                    z=float(rec['z']),
                    u=float(rec['u']),
                    v=float(rec['v']),
                    lat=float(rec['lat']),
                    lon=float(rec['lon']),
                )
                try:
                    del partial[rec_key]
                except Exception as err:
                    print('handle message error:', err)
                    pass
            else:
                pass
                #print('handler has not received:', list(set(('z', 'u', 'v', 'lat', 'lon')) - set(rec.keys())))


            while len(partial) > 100:
                try:
                    partial.popitem(last=False)
                except Exception as err:
                    print('handle message error:', err)
                    break
