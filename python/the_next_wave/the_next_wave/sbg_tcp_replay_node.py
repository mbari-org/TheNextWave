#!/usr/bin/env python3

"""SBG TCP replay node.

Reads a raw SBG MATLAB file (same format used by `example.py`) and replays a
minimal subset of SBG ECom binary frames to a TCP server (Ethernet bridge).

This is intended for local testing of the TCP ingest path.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from the_next_wave import sbgMessageParse
from the_next_wave.download_example_data import get_example_data_dir
from the_next_wave.utilities import load_raw_sbg_arrays, select_sbg_burst_struct

try:
    import scipy.io as spio
except ImportError as e:  # pragma: no cover
    spio = None
    SCIPY_IMPORT_ERROR = e


SYNC = b"\xff\x5a"
MSG_CLASS_ECOM_LOG = b"\x00"
ETX = b"\x33"
DUMMY_CRC = b"\x00\x00"


def loadmat_struct(path: str | Path):
    if spio is None:  # pragma: no cover
        raise ImportError("scipy is required to load MATLAB .mat files") from SCIPY_IMPORT_ERROR
    return spio.loadmat(str(path), struct_as_record=False, squeeze_me=True)


class SbgTcpReplayNode(Node):
    def __init__(self):
        super().__init__("the_next_wave_sbg_tcp_replay")

        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 3001)
        self.declare_parameter("sbg_mat_path", "")
        self.declare_parameter("swift_num", 22)
        self.declare_parameter("start_index", 0)
        self.declare_parameter("end_index", -1)
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("connect_retry_sec", 1.0)

        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.sbg_mat_path = str(self.get_parameter("sbg_mat_path").value)
        self.swift_num = int(self.get_parameter("swift_num").value)
        self.start_index = int(self.get_parameter("start_index").value)
        self.end_index = int(self.get_parameter("end_index").value)
        self.speed = float(self.get_parameter("speed").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.connect_retry_sec = float(self.get_parameter("connect_retry_sec").value)
        if not np.isfinite(self.connect_retry_sec) or self.connect_retry_sec <= 0.0:
            self.connect_retry_sec = 1.0

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def resolve_default_mat_path(self) -> str:
        # Use/download the pinned example dataset.
        example_dir = get_example_data_dir()
        if not example_dir.is_dir():
            raise FileNotFoundError(f"Example data dir does not exist: {example_dir}")

        # Prefer the known SWIFT file names used in our launch defaults.
        preferred = example_dir / f"SWIFT{self.swift_num}_SBG_12Sep2022_07_01.mat"
        if preferred.is_file():
            return str(preferred)

        # Otherwise fall back to the first matching file.
        matches = sorted(example_dir.glob(f"SWIFT{self.swift_num}_SBG_*.mat"))
        if matches:
            return str(matches[0])

        raise FileNotFoundError(
            f"No SWIFT{self.swift_num} SBG .mat files found in example data dir: {example_dir}"
        )

    def destroy_node(self):
        try:
            self.stop_event.set()
        except Exception:
            pass
        return super().destroy_node()

    def send_msg(self, sock: socket.socket, msg_id: bytes, values: tuple) -> None:
        info = sbgMessageParse.sbgMessages.get(msg_id)
        if info is None:
            raise ValueError(f"Unknown msg_id {msg_id!r}")

        payload = struct.pack(info["unpackString"], *values)
        if len(payload) != int(info["intLength"]):
            raise ValueError("Packed payload length mismatch")

        frame = SYNC + msg_id + MSG_CLASS_ECOM_LOG + info["binLength"] + payload + DUMMY_CRC + ETX
        sock.sendall(frame)

    def run_once(self) -> None:
        if not self.sbg_mat_path:
            self.sbg_mat_path = self.resolve_default_mat_path()
            self.get_logger().info(f"Using SBG .mat: {self.sbg_mat_path}")

        mat = loadmat_struct(self.sbg_mat_path)
        sbg_data = mat["sbgData"]
        sbg = select_sbg_burst_struct(sbg_data, prefer_longest=True)
        if getattr(sbg_data, "size", 1) > 1:
            self.get_logger().info("sbgData contains multiple bursts; selected the longest")

        t_us, heave, vel_e, vel_n, lat, lon = load_raw_sbg_arrays(
            sbg,
            start_index=self.start_index,
            end_index=self.end_index,
        )

        # Connect (retry until success or shutdown)
        sock: socket.socket | None = None
        while rclpy.ok() and not self.stop_event.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                break
            except OSError:
                time.sleep(self.connect_retry_sec)

        if sock is None:
            return

        self.get_logger().info(f"Connected to SBG bridge at {self.host}:{self.port}")

        try:
            last_t = None
            for i in range(int(t_us.size)):
                if self.stop_event.is_set():
                    break

                # Use uint32 for the on-wire time_stamp field.
                ts = int(np.asarray(t_us[i]).item()) & 0xFFFFFFFF

                # ShipMotion (0x09) format: <L10fH
                shipmotion = (
                    ts,
                    0.0,  # heave_period
                    0.0,  # surge
                    0.0,  # sway
                    float(np.asarray(heave[i]).item()),  # heave
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0,  # heave_status
                )
                self.send_msg(sock, b"\x09", shipmotion)

                # GpsVel (0x0d) format: <LLL8f
                gpsvel = (
                    ts,
                    0,  # gps_vel_status
                    0,  # gps_tow
                    float(np.asarray(vel_n[i]).item()),  # vel_n
                    float(np.asarray(vel_e[i]).item()),  # vel_e
                    0.0,  # vel_d
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # course
                    0.0,  # course_acc
                )
                self.send_msg(sock, b"\x0d", gpsvel)

                # GpsPos (0x0e) format: <LLL3d4fBHH
                gpspos = (
                    ts,
                    0,  # gps_pos_status
                    0,  # gps_tow
                    float(np.asarray(lat[i]).item()),
                    float(np.asarray(lon[i]).item()),
                    0.0,  # altitude
                    0.0,  # undulation
                    0.0,
                    0.0,
                    0.0,
                    0,  # num_sv_used
                    0,  # base_station_id
                    0,  # diff_age
                )
                self.send_msg(sock, b"\x0e", gpspos)

                # Sleep based on timestamps (scaled)
                t_now = int(np.asarray(t_us[i]).item())
                if last_t is not None and self.speed > 0.0:
                    dt_s = max(0.0, (t_now - last_t) / 1e6)
                    time.sleep(dt_s / float(self.speed))
                last_t = t_now

        finally:
            try:
                sock.close()
            except Exception:
                pass

    def run(self) -> None:
        while rclpy.ok() and not self.stop_event.is_set():
            try:
                self.run_once()
            except Exception as e:
                self.get_logger().error(f"SBG TCP replay failed: {e}")
                time.sleep(1.0)

            if not self.loop:
                break


def main(args=None):
    rclpy.init(args=args)
    node = SbgTcpReplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
