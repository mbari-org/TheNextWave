#!/usr/bin/env python3

"""
Block until the Gazebo ``/clock`` has advanced a requested number of sim seconds.

Used by ``scripts/run_sim_cases.sh`` to size a run by simulated duration rather
than wall-clock, since Gazebo real-time factor varies. Exits non-zero if the
clock never starts, stalls, or the wall-clock guard expires.
"""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock


CLOCK_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
)


class ClockWaiter(Node):
    """Tracks /clock and reports elapsed sim time since the first sample."""

    def __init__(self, duration_sec: float, stall_timeout_sec: float):
        super().__init__('wait_sim_time')
        self.duration_sec = float(duration_sec)
        self.stall_timeout_sec = float(stall_timeout_sec)
        self.t_start = None
        self.t_now = None
        self.last_advance_walltime = time.monotonic()
        self.create_subscription(Clock, '/clock', self.clock_callback, CLOCK_QOS)

    def clock_callback(self, msg: Clock) -> None:
        t = float(msg.clock.sec) + float(msg.clock.nanosec) * 1e-9
        if self.t_start is None:
            self.t_start = t
            self.get_logger().info(f'sim clock started at {t:.3f} s')
        elif t < self.t_start:
            # Sim reset; re-baseline rather than waiting forever.
            self.get_logger().warn(f'sim clock went backwards to {t:.3f} s; re-baselining')
            self.t_start = t
        if self.t_now is None or t > self.t_now:
            self.last_advance_walltime = time.monotonic()
        self.t_now = t

    @property
    def elapsed(self) -> float:
        if self.t_start is None or self.t_now is None:
            return 0.0
        return self.t_now - self.t_start

    def done(self) -> bool:
        return self.elapsed >= self.duration_sec

    def stalled(self) -> bool:
        if self.t_start is None:
            return False
        return (time.monotonic() - self.last_advance_walltime) > self.stall_timeout_sec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, required=True,
                        help='sim seconds to wait for after the first /clock message')
    parser.add_argument('--startup-timeout', type=float, default=180.0,
                        help='wall seconds to wait for the first /clock message')
    parser.add_argument('--stall-timeout', type=float, default=120.0,
                        help='wall seconds of no clock advance before giving up')
    parser.add_argument('--wall-timeout', type=float, default=0.0,
                        help='hard wall-clock cap in seconds (0 = none)')
    parser.add_argument('--progress-every', type=float, default=30.0,
                        help='sim seconds between progress lines')
    args = parser.parse_args()

    rclpy.init()
    node = ClockWaiter(args.duration, args.stall_timeout)
    wall_start = time.monotonic()
    next_report = args.progress_every
    status = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            wall = time.monotonic() - wall_start

            if node.t_start is None:
                if wall > args.startup_timeout:
                    node.get_logger().error(
                        f'no /clock within {args.startup_timeout:.0f}s wall time'
                    )
                    status = 2
                    break
                continue

            if node.elapsed >= next_report:
                node.get_logger().info(
                    f'sim elapsed {node.elapsed:.1f}/{args.duration:.1f} s '
                    f'(wall {wall:.1f} s, RTF~{node.elapsed / max(wall, 1e-6):.2f})'
                )
                next_report += args.progress_every

            if node.done():
                node.get_logger().info(
                    f'reached {node.elapsed:.1f} sim s in {wall:.1f} wall s'
                )
                break
            if node.stalled():
                node.get_logger().error(
                    f'/clock stalled at {node.elapsed:.1f} sim s '
                    f'(no advance for {args.stall_timeout:.0f}s)'
                )
                status = 3
                break
            if args.wall_timeout > 0.0 and wall > args.wall_timeout:
                node.get_logger().error(
                    f'wall-clock cap {args.wall_timeout:.0f}s hit at '
                    f'{node.elapsed:.1f} sim s'
                )
                status = 4
                break
    except KeyboardInterrupt:
        status = 130
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return status


if __name__ == '__main__':
    sys.exit(main())
