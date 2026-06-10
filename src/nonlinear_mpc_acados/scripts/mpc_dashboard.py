#!/usr/bin/env python3
"""Live MPCC telemetry dashboard — a "instrument panel" window.

Subscribes to /mpc_debug (Float32MultiArray, DBG_FIELDS order) and shows:
  - speed: v_actual / v_cmd / ref_v (rolling line)
  - solver: solve_ms (rolling line + big readout)
  - cost: opti_value (rolling line)
  - steer + side_pref
  - big text gauges (v, solve_ms, cost, v_max_cost, kappa, s, feas-ish)

Usage:  python3 mpc_dashboard.py        (run alongside a sim)
Needs a display ($DISPLAY). Read-only; does not affect the controller.
"""
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# /mpc_debug field order — must match mpc_debug_logger.DBG_FIELDS
F = {n: i for i, n in enumerate([
    "v_cmd", "steer_cmd", "v_actual", "car_x", "car_y", "car_yaw",
    "current_s", "near_idx", "ref_v", "n_obs_in", "sel_dmin", "sel_x", "sel_y",
    "side_pref", "opti_value", "solve_ms", "kappa_abs", "kappa_signed",
    "q_cte_scale", "q_lag_scale", "q_v_scale", "q_drate_scale", "v_max_cost",
])}

N = 300  # rolling window samples


class DashSub(Node):
    def __init__(self):
        super().__init__('mpc_dashboard')
        self.lock = threading.Lock()
        self.hist = {k: deque(maxlen=N) for k in
                     ('v_actual', 'v_cmd', 'ref_v', 'solve_ms', 'opti_value',
                      'steer_cmd', 'side_pref')}
        self.latest = {}
        self.create_subscription(Float32MultiArray, '/mpc_debug', self._cb, 10)

    def _cb(self, msg):
        d = msg.data
        if len(d) <= F['v_max_cost']:
            return
        with self.lock:
            for k in self.hist:
                self.hist[k].append(float(d[F[k]]))
            self.latest = {k: float(d[F[k]]) for k in F if F[k] < len(d)}


def main():
    rclpy.init()
    node = DashSub()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    fig = plt.figure(figsize=(11, 7))
    fig.canvas.manager.set_window_title('MPCC Telemetry')
    gs = fig.add_gridspec(3, 2)
    ax_v = fig.add_subplot(gs[0, 0])
    ax_solve = fig.add_subplot(gs[1, 0])
    ax_cost = fig.add_subplot(gs[2, 0])
    ax_steer = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1:, 1]); ax_txt.axis('off')
    fig.tight_layout(pad=2.0)

    def draw(_):
        with node.lock:
            h = {k: list(v) for k, v in node.hist.items()}
            L = dict(node.latest)
        ax_v.clear(); ax_solve.clear(); ax_cost.clear(); ax_steer.clear(); ax_txt.clear(); ax_txt.axis('off')
        if h['v_actual']:
            x = range(len(h['v_actual']))
            ax_v.plot(x, h['v_actual'], 'b-', label='v_actual')
            ax_v.plot(x, h['v_cmd'], 'g-', lw=0.8, label='v_cmd')
            ax_v.plot(range(len(h['ref_v'])), h['ref_v'], 'r--', lw=0.8, label='ref_v')
            ax_v.set_ylabel('speed [m/s]'); ax_v.legend(loc='upper left', fontsize=7); ax_v.grid(alpha=0.3)
            ax_solve.plot(range(len(h['solve_ms'])), h['solve_ms'], 'm-')
            ax_solve.axhline(40, color='r', ls=':', lw=0.8)  # 25Hz budget
            ax_solve.set_ylabel('solve [ms]'); ax_solve.grid(alpha=0.3)
            ax_cost.plot(range(len(h['opti_value'])), h['opti_value'], 'c-')
            ax_cost.set_ylabel('cost'); ax_cost.set_xlabel('samples'); ax_cost.grid(alpha=0.3)
            ax_steer.plot(range(len(h['steer_cmd'])), h['steer_cmd'], 'k-')
            ax_steer.set_ylabel('steer [rad]'); ax_steer.set_ylim(-0.5, 0.5); ax_steer.grid(alpha=0.3)
        if L:
            txt = (f"v        {L.get('v_actual',0):5.2f} m/s\n"
                   f"v_cmd    {L.get('v_cmd',0):5.2f} m/s\n"
                   f"ref_v    {L.get('ref_v',0):5.2f} m/s\n"
                   f"v_cap    {L.get('v_max_cost',0):5.2f} m/s\n"
                   f"─────────────\n"
                   f"solve    {L.get('solve_ms',0):5.1f} ms\n"
                   f"cost     {L.get('opti_value',0):7.1f}\n"
                   f"steer    {L.get('steer_cmd',0):+5.3f}\n"
                   f"|κ|fwd   {L.get('kappa_abs',0):5.2f}\n"
                   f"s        {L.get('current_s',0):5.1f} m\n"
                   f"obs      {int(L.get('n_obs_in',0))}  side {int(L.get('side_pref',0)):+d}")
            ax_txt.text(0.05, 0.95, txt, family='monospace', fontsize=14,
                        va='top', transform=ax_txt.transAxes)
    ani = FuncAnimation(fig, draw, interval=100, cache_frame_data=False)
    try:
        plt.show()
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
