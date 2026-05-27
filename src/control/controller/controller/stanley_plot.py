#!/usr/bin/env python3
"""Real-time plot of Stanley controller errors.

Usage:
  ros2 run controller stanley_plot
  or
  python3 stanley_plot.py

Plots (sliding window):
  - Cross-track error  /stanley/cte
  - Heading error      /stanley/heading_error
  - Steering command   /stanley/steer
"""
from __future__ import annotations

import collections
import threading

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

WINDOW  = 10.0   # seconds of history shown
HZ      = 50.0   # expected publish rate
MAXLEN  = int(WINDOW * HZ)

CTE_YLIM     = (-0.5,  0.5)   # [m]
PSI_YLIM     = (-0.5,  0.5)   # [rad]
STEER_YLIM   = (-0.45, 0.45)  # [rad]


class _Buf:
    def __init__(self):
        self.t   = collections.deque(maxlen=MAXLEN)
        self.val = collections.deque(maxlen=MAXLEN)
        self.t0  = None

    def push(self, v: float, stamp_sec: float):
        if self.t0 is None:
            self.t0 = stamp_sec
        self.t.append(stamp_sec - self.t0)
        self.val.append(v)


class StanleyPlotNode(Node):
    def __init__(self):
        super().__init__("stanley_plot")
        self.cte     = _Buf()
        self.heading = _Buf()
        self.steer   = _Buf()
        self._lock   = threading.Lock()

        now = lambda: self.get_clock().now().nanoseconds * 1e-9

        def _cb(buf):
            def cb(msg: Float64):
                with self._lock:
                    buf.push(msg.data, now())
            return cb

        self.create_subscription(Float64, "/stanley/cte",           _cb(self.cte),     10)
        self.create_subscription(Float64, "/stanley/heading_error", _cb(self.heading), 10)
        self.create_subscription(Float64, "/stanley/steer",         _cb(self.steer),   10)


def main(args=None):
    rclpy.init(args=args)
    node = StanleyPlotNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
    fig.suptitle("Stanley controller — real-time errors", fontsize=13)

    configs = [
        (node.cte,     axes[0], "CTE [m]",            CTE_YLIM,   "tab:blue"),
        (node.heading, axes[1], "Heading error [rad]", PSI_YLIM,   "tab:orange"),
        (node.steer,   axes[2], "Steer cmd [rad]",     STEER_YLIM, "tab:green"),
    ]

    lines = []
    zero_lines = []
    for buf, ax, ylabel, ylim, color in configs:
        (line,) = ax.plot([], [], color=color, linewidth=1.2)
        zl = ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        lines.append(line)
        zero_lines.append(zl)

    axes[-1].set_xlabel("time [s]")

    def _update(_frame):
        with node._lock:
            for (buf, ax, *_), line in zip(configs, lines):
                t   = list(buf.t)
                val = list(buf.val)
                if not t:
                    continue
                t_max = t[-1]
                t_min = max(0.0, t_max - WINDOW)
                line.set_data(t, val)
                ax.set_xlim(t_min, t_min + WINDOW)
        return lines

    ani = animation.FuncAnimation(fig, _update, interval=100, blit=False)
    plt.tight_layout()
    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
