#!/usr/bin/env python3
"""f map prep for EVO-MPCC base.

Decisions per user spec:
  - Raceline = centerline (the conservative middle line, no IQP). The MPC
    will track the centerline as its reference line.
  - ref_v: constant max_speed (kept simple — no per-point velocity profile).
  - Track bounds: uniform 1.0 m on each side (d_left = d_right = 1.0).
    This is independent of the d_left/d_right stored in the json so the
    boundary stays a clean ±1 m halo around the centerline regardless of
    map noise. Hairpin self-intersect is avoided as long as the tightest
    radius (1/κ_max) > 1.0 m. Verified below.

Output (overwriting trackf_*.csv that the EVO-MPCC node reads):
  trackf_centerline_waypoints.csv : x, y, ref_v
  trackf_left_waypoints.csv       : x, y    (centerline + 1.0·n)
  trackf_right_waypoints.csv      : x, y    (centerline − 1.0·n)
  trackf_center_derivates.csv     : dx, dy  (forward finite diff, closed loop)
"""
import json
import csv
import math
import numpy as np

SRC = '/home/hmcl/unicorn-racing-stack/stack_master/maps/f/global_waypoints.json'
OUT = '/home/hmcl/unicorn-racing-stack/evo_mpcc/toolkit/tracks/trackf/'

# ---- Tunables (per-user) ----
HALF_BOUND  = 1.0     # [m] uniform half-track width (left = right = 1.0)
V_MAX       = 6.0     # [m/s] cap on straights (matches yaml max_speed)
V_MIN       = 1.5     # [m/s] floor (avoid stalling at sharpest hairpins)
A_LAT_MAX   = 6.0     # [m/s²] lateral-accel limit → defines v(κ)
KAPPA_SMOOTH = 25     # bigger window → smoother v_ref transitions in/out of corners

with open(SRC) as f:
    data = json.load(f)
cl = data['centerline_waypoints']['wpnts']
n = len(cl)

# Sanity check: tightest centerline radius vs HALF_BOUND
kappas = np.array([abs(p['kappa_radpm']) for p in cl])
min_radius = 1.0 / kappas.max() if kappas.max() > 0 else float('inf')
print(f"centerline pts: {n}")
print(f"min curvature radius: {min_radius:.3f} m  (HALF_BOUND={HALF_BOUND})")
if HALF_BOUND >= min_radius:
    print(f"WARNING: HALF_BOUND >= min radius → boundary will self-intersect "
          f"at the sharpest corner. Consider HALF_BOUND < {min_radius:.2f}.")

# --- centerline + curvature-based ref_v ---
# ref_v(s) = clip( sqrt(A_LAT_MAX / |κ_smooth(s)|),  V_MIN, V_MAX )
# Smooth κ first to avoid v_ref jitter that would feed steering oscillation.
kappa = np.array([abs(p['kappa_radpm']) for p in cl])
def circ_smooth(a, w):
    pad = np.concatenate([a[-w:], a, a[:w]])
    return np.convolve(pad, np.ones(w)/w, mode='same')[w:-w]
kappa_sm = circ_smooth(kappa, KAPPA_SMOOTH) if KAPPA_SMOOTH > 1 else kappa
v_ref_arr = np.clip(np.sqrt(A_LAT_MAX / (kappa_sm + 1e-3)), V_MIN, V_MAX)
print(f"ref_v profile: min={v_ref_arr.min():.2f}  max={v_ref_arr.max():.2f}  "
      f"mean={v_ref_arr.mean():.2f} m/s")

with open(OUT + 'trackf_centerline_waypoints.csv', 'w', newline='') as fc:
    wc = csv.writer(fc)
    for i, p in enumerate(cl):
        wc.writerow([f"{p['x_m']:.6f}", f"{p['y_m']:.6f}", f"{v_ref_arr[i]:.6f}"])
print("wrote trackf_centerline_waypoints.csv  (raceline = centerline, "
      f"ref_v ∈ [{V_MIN}, {V_MAX}] curvature-based)")

# --- left/right (uniform halo) + tangent derivatives ---
with open(OUT + 'trackf_left_waypoints.csv',  'w', newline='') as fl, \
     open(OUT + 'trackf_right_waypoints.csv', 'w', newline='') as fr, \
     open(OUT + 'trackf_center_derivates.csv', 'w', newline='') as fd:
    wl = csv.writer(fl); wr = csv.writer(fr); wd = csv.writer(fd)
    for i, p in enumerate(cl):
        x, y, psi = p['x_m'], p['y_m'], p['psi_rad']
        nx, ny = -math.sin(psi), math.cos(psi)
        wl.writerow([f"{x + HALF_BOUND * nx:.6f}", f"{y + HALF_BOUND * ny:.6f}"])
        wr.writerow([f"{x - HALF_BOUND * nx:.6f}", f"{y - HALF_BOUND * ny:.6f}"])
        nxt = cl[(i + 1) % n]
        wd.writerow([f"{nxt['x_m'] - x:.6f}", f"{nxt['y_m'] - y:.6f}"])
print(f"wrote trackf_left/right_waypoints.csv  (uniform ±{HALF_BOUND} m)")
print("wrote trackf_center_derivates.csv  (forward diff, closed loop)")
