#!/usr/bin/env python3
"""gen_random_track.py — random Voronoi 트랙을 F1Tenth/stack_master 포맷으로 변환.

흐름:
  1. random_track_generator._create_track() 로 cones_left/right + centerline 1000pt 생성
  2. centerline 을 uniform arc-length (default 0.1m) 로 재샘플 → s_m,x_m,y_m,psi,kappa
  3. vx_mps = vmax · clip(1 - |κ|/κ_safe, vmin_frac, 1) — corner-aware ref
  4. PNG occupancy grid 생성 (트랙 polygon 안 free, 바깥 occupied)
  5. <name>.yaml + global_waypoints.json + ot_sectors.yaml + speed_scaling.yaml 저장

Usage:
  python3 gen_random_track.py --name rand01 --seed 7 --preset small --vmax 6.0
  python3 gen_random_track.py --name rand02 --seed 11 --preset medium --width 3.5

Output:
  src/stack_master/maps/<name>/
    ├── <name>.yaml
    ├── <name>.png
    ├── global_waypoints.json
    ├── ot_sectors.yaml
    └── speed_scaling.yaml
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import interpolate
from shapely.geometry import LineString, Polygon

REPO_ROOT = Path(__file__).resolve().parents[3]
RTG_PATH = REPO_ROOT / "random-track-generator"
if str(RTG_PATH) not in sys.path:
    sys.path.insert(0, str(RTG_PATH))

from random_track_generator.track_generator import _create_track  # noqa: E402
from random_track_generator.track import Mode  # noqa: E402

PRESETS = {
    # F1Tenth BO sweet-spot: **75~110m centerline + width 2.5m** (narrow corridor).
    # 2026-05-26 사용자 요청 + 100-seed sweep 으로 calibrate.
    # 같은 config 의 seed 별 L distribution 이 넓어 — 권장 seed 16/28/37/41/53/62.
    "tiny":   dict(n_points=10, n_regions=5,  min_bound=0., max_bound=30., mode=Mode.EXTEND),
    # **race**: BO 학습용 표준 preset. 80~100m + hairpin (f map 같은 복잡), width 2.5.
    # 2026-05-27 사용자 요청: f map 같은 복잡 + 2.5m 너비.
    "race":   dict(n_points=22, n_regions=9,  min_bound=0., max_bound=30., mode=Mode.RANDOM),
    # complex (100~150m hairpin) — generalization 검증 용.
    "complex":dict(n_points=30, n_regions=12, min_bound=0., max_bound=50., mode=Mode.RANDOM),
    # legacy.
    "short":  dict(n_points=22, n_regions=9,  min_bound=0., max_bound=30., mode=Mode.RANDOM),
    "mid":    dict(n_points=22, n_regions=9,  min_bound=0., max_bound=30., mode=Mode.RANDOM),
    # generalization stress test 용 (200m+).
    "huge":   dict(n_points=35, n_regions=15, min_bound=0., max_bound=80., mode=Mode.RANDOM),
}


def gen_track_with_retry(preset: str, seed: int, max_tries: int = 20):
    p = dict(PRESETS[preset])
    for k in range(max_tries):
        try:
            return _create_track(seed=seed + 1000 * k, **p)
        except Exception as e:  # noqa: BLE001
            print(f"  [retry {k+1}] seed={seed + 1000*k} failed: {e}")
    raise RuntimeError(f"Could not generate track after {max_tries} tries")


def resample_uniform(centerline: np.ndarray, ds: float = 0.1) -> np.ndarray:
    """centerline (N,2) → uniform arc-length sample. Returns (M,2), s, total_length."""
    # Make sure path closes.
    pts = np.vstack([centerline, centerline[0:1]]) if not np.allclose(centerline[0], centerline[-1]) else centerline.copy()
    seg = np.diff(pts, axis=0)
    L = np.cumsum(np.linalg.norm(seg, axis=1))
    L = np.insert(L, 0, 0.0)
    total = float(L[-1])
    # Periodic spline parameterized in [0,1].
    u = L / total
    tck, _ = interpolate.splprep([pts[:, 0], pts[:, 1]], u=u, s=0, per=True)
    n = max(int(round(total / ds)), 64)
    s_new = np.linspace(0.0, total, n, endpoint=False)
    x, y = interpolate.splev(s_new / total, tck)
    return np.column_stack([x, y]), s_new, total


def forward_backward_vel(kappa: np.ndarray, ds: float, vmax: float,
                         a_lat_max: float = 8.0, a_long_max: float = 4.0,
                         a_brake_max: float = 6.0, n_iter: int = 3) -> np.ndarray:
    """Heilmeier-style forward-backward velocity profile (brake-aware ref_v).

    Algorithm (matches trajectory_planning_helpers/calc_vel_profile.py):
      1. v_kin(s) = sqrt(a_lat_max / |κ(s)|)   — lateral grip cap (point-mass)
      2. v(s) = min(vmax, v_kin(s))            — global vmax cap
      3. Loop n_iter times (periodic, closed loop):
         a. forward pass: v[i+1]² ≤ v[i]² + 2·a_long_max·ds   — accel limit
         b. backward pass: v[i]² ≤ v[i+1]² + 2·a_brake_max·ds — brake limit

    Effect: at corner exit ref_v ramps UP gradually (accel limit), and on
    corner approach ref_v ramps DOWN early (brake distance). Short MPC horizon
    can still "see" the upcoming brake point because it's baked into ref_v(s).

    Args:
      kappa:    (N,) absolute curvature [1/m]
      ds:       arc-length step [m] (uniform)
      vmax:     global max speed [m/s]
      a_lat_max:  lateral grip [m/s²]  (≈ μ·g ≈ 8 for F1Tenth)
      a_long_max: forward accel cap [m/s²]
      a_brake_max: braking decel cap [m/s²] (≥ a_long_max typically)
      n_iter: forward+backward loop count (3 enough for periodic convergence)

    Returns:
      v: (N,) ref velocity profile [m/s]
    """
    n = len(kappa)
    abs_k = np.abs(kappa) + 1e-6
    v_kin = np.sqrt(a_lat_max / abs_k)
    v = np.minimum(vmax, v_kin)

    for _ in range(n_iter):
        # Forward pass — acceleration limit
        for i in range(n - 1):
            cap = np.sqrt(v[i] ** 2 + 2.0 * a_long_max * ds)
            if cap < v[i + 1]:
                v[i + 1] = cap
        cap = np.sqrt(v[-1] ** 2 + 2.0 * a_long_max * ds)
        if cap < v[0]:
            v[0] = cap     # periodic wrap

        # Backward pass — brake limit
        for i in range(n - 1, 0, -1):
            cap = np.sqrt(v[i] ** 2 + 2.0 * a_brake_max * ds)
            if cap < v[i - 1]:
                v[i - 1] = cap
        cap = np.sqrt(v[0] ** 2 + 2.0 * a_brake_max * ds)
        if cap < v[-1]:
            v[-1] = cap    # periodic wrap

    return v


def compute_psi_kappa(xy: np.ndarray, total_length: float):
    """Heading + curvature on a closed-loop centerline."""
    n = len(xy)
    # Wrap for finite diff
    x = xy[:, 0]
    y = xy[:, 1]
    xp = np.roll(x, -1) - np.roll(x, 1)
    yp = np.roll(y, -1) - np.roll(y, 1)
    # Arc-length step
    ds = total_length / n
    dx = xp / (2 * ds)
    dy = yp / (2 * ds)
    d2x = (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / (ds * ds)
    d2y = (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / (ds * ds)
    psi = np.arctan2(dy, dx)
    denom = (dx * dx + dy * dy) ** 1.5
    denom = np.where(denom < 1e-6, 1e-6, denom)
    kappa = (dx * d2y - dy * d2x) / denom
    return psi, kappa


def render_occupancy(track_poly: Polygon, resolution: float = 0.05, margin: float = 1.5):
    """Polygon → PIL image + yaml metadata. White=free, black=occupied."""
    minx, miny, maxx, maxy = track_poly.bounds
    minx -= margin
    miny -= margin
    maxx += margin
    maxy += margin
    width_px = max(int(math.ceil((maxx - minx) / resolution)), 1)
    height_px = max(int(math.ceil((maxy - miny) / resolution)), 1)
    img = Image.new("L", (width_px, height_px), 0)  # 0 = occupied
    draw = ImageDraw.Draw(img)

    def world_to_px(x, y):
        # ROS map: origin = world coord of bottom-left pixel; image y is flipped.
        px = (x - minx) / resolution
        py = height_px - (y - miny) / resolution
        return (px, py)

    exterior = [world_to_px(x, y) for x, y in track_poly.exterior.coords]
    draw.polygon(exterior, fill=254)  # 254 = free
    for interior in track_poly.interiors:
        draw.polygon([world_to_px(x, y) for x, y in interior.coords], fill=0)

    origin = [float(minx), float(miny), 0.0]
    return img, origin, resolution


def make_global_waypoints(centerline_xy: np.ndarray, psi, kappa, total_length, width: float,
                          vmax: float, a_lat_max: float = 8.0, a_long_max: float = 4.0,
                          a_brake_max: float = 6.0) -> dict:
    n = len(centerline_xy)
    ds = total_length / n
    s_m = np.arange(n, dtype=float) * ds

    # Brake-aware reference velocity profile (Heilmeier forward-backward).
    # Replaces old point-mass: vx = vmax · clip(1 - |κ|/κ_safe, vmin, 1).
    # New: bake longitudinal accel/brake limits so short MPC horizon can see
    # upcoming brake point.
    vx = forward_backward_vel(kappa, ds=ds, vmax=vmax,
                              a_lat_max=a_lat_max,
                              a_long_max=a_long_max,
                              a_brake_max=a_brake_max)
    ax = np.gradient(vx) / ds
    half = width / 2.0

    def wpnt(i):
        return {
            "id": int(i),
            "s_m": float(s_m[i]),
            "d_m": 0.0,
            "x_m": float(centerline_xy[i, 0]),
            "y_m": float(centerline_xy[i, 1]),
            "d_right": float(half),
            "d_left": float(half),
            "psi_rad": float(psi[i]),
            "kappa_radpm": float(kappa[i]),
            "vx_mps": float(vx[i]),
            "ax_mps2": float(ax[i]),
        }

    centerline_wpnts = [wpnt(i) for i in range(n)]
    # IQP / SP = centerline copy with vx (we don't run IQP for random tracks).
    cl_wpnts_with_v0 = [{**w, "vx_mps": 0.0, "ax_mps2": 0.0} for w in centerline_wpnts]

    est_lap = float(np.sum(ds / vx))
    info = (f"Random track. length={total_length:.2f}m, vmax={vmax:.2f}m/s, "
            f"est_lap_time={est_lap:.2f}s; SP maximum speed: {vmax:.2f}m/s;")
    empty_markers = {"markers": []}

    return {
        "map_info_str": {"data": info},
        "est_lap_time": {"data": est_lap},
        "centerline_markers": empty_markers,
        "centerline_waypoints": {
            "header": {"seq": 1, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""},
            "wpnts": cl_wpnts_with_v0,
        },
        "global_traj_markers_iqp": empty_markers,
        "global_traj_wpnts_iqp": {
            "header": {"seq": 1, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""},
            "wpnts": centerline_wpnts,
        },
        "global_traj_markers_sp": empty_markers,
        "global_traj_wpnts_sp": {
            "header": {"seq": 1, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""},
            "wpnts": centerline_wpnts,
        },
        "trackbounds_markers": empty_markers,
    }


def write_csv_track(csv_dir: Path, name: str, centerline_xy: np.ndarray, psi, kappa,
                    width: float, vmax: float, total_length: float,
                    a_lat_max: float = 8.0, a_long_max: float = 4.0,
                    a_brake_max: float = 6.0):
    """Emit the CSV bundle that nonlinear_mpc_acados/track_loader expects.

    File layout: <csv_dir>/track<name>/track<name>_*.csv
      - track<name>_centerline_waypoints.csv   x, y, ref_v
      - track<name>_center_derivates.csv       dx, dy   (unit tangent)
      - track<name>_right_waypoints.csv        x, y     (centerline + half·n)
      - track<name>_left_waypoints.csv         x, y     (centerline - half·n)
    """
    sub = csv_dir / f'track{name}'
    sub.mkdir(parents=True, exist_ok=True)
    base = sub / f'track{name}'

    half = width / 2.0
    # tangent unit vector from heading psi
    tx = np.cos(psi)
    ty = np.sin(psi)
    # left normal n_l = (-ty, tx); right normal n_r = (ty, -tx).
    # _create_track sorts the centerline clockwise — apply convention used by
    # the f-track CSV (left side is +y of forward heading).
    left = centerline_xy + np.column_stack([-ty, tx]) * half
    right = centerline_xy + np.column_stack([ty, -tx]) * half

    # Reference speed — Heilmeier forward-backward (same as JSON wpnts).
    ds = total_length / len(centerline_xy)
    vx = forward_backward_vel(kappa, ds=ds, vmax=vmax,
                              a_lat_max=a_lat_max,
                              a_long_max=a_long_max,
                              a_brake_max=a_brake_max)

    def _save(p: Path, arr: np.ndarray):
        np.savetxt(p, arr, delimiter=',', fmt='%.6f')

    _save(base.with_name(f'track{name}_centerline_waypoints.csv'),
          np.column_stack([centerline_xy[:, 0], centerline_xy[:, 1], vx]))
    _save(base.with_name(f'track{name}_center_derivates.csv'),
          np.column_stack([tx, ty]))
    _save(base.with_name(f'track{name}_right_waypoints.csv'), right)
    _save(base.with_name(f'track{name}_left_waypoints.csv'), left)
    return sub


def write_track(out_dir: Path, name: str, img: Image.Image, origin, resolution,
                gw: dict, width: float, est_lap: float, start_psi: float = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    # PNG
    img_path = out_dir / f"{name}.png"
    img.save(img_path)
    # map yaml
    map_yaml = {
        "image": f"{name}.png",
        "resolution": float(resolution),
        "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    import yaml as _yaml
    (out_dir / f"{name}.yaml").write_text(_yaml.safe_dump(map_yaml, sort_keys=False))
    (out_dir / "global_waypoints.json").write_text(json.dumps(gw))
    # Minimal sector/speed-scaling so other nodes don't crash.
    (out_dir / "ot_sectors.yaml").write_text(_yaml.safe_dump({"ot_sectors": []}))
    (out_dir / "speed_scaling.yaml").write_text(_yaml.safe_dump({"speed_scaling": []}))
    # Spawn at centerline[0] (≈ origin after the roll above) facing the
    # centerline tangent there. _create_track's intended start_heading=π/2
    # is unreliable because the spline parameterization is independent of
    # the start-pose alignment — we read the actual tangent from psi[0].
    stheta = float(start_psi) if start_psi is not None else float(np.pi / 2)
    (out_dir / "start_pose.yaml").write_text(_yaml.safe_dump({
        "sx": 0.0, "sy": 0.0, "stheta": stheta,
        "sx1": 1.0 * float(np.cos(stheta)),
        "sy1": 1.0 * float(np.sin(stheta)),
        "stheta1": stheta,
    }))
    # global_planner.trajectory_optimizer 입력 (centerline_extractor 건너뛰기 위해).
    # Format: x_m, y_m, w_tr_right_m, w_tr_left_m (CSV with header). 우리 centerline
    # 이 (0,0) 시작 + roll 정확하므로 extract 의 mismatch (spawn 90° off) 방지.
    import csv as _csv
    cl_wpnts = gw["centerline_waypoints"]["wpnts"]
    with open(out_dir / "centerline.csv", "w", newline="") as _f:
        _w = _csv.writer(_f)
        _w.writerow(["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
        for wp in cl_wpnts:
            _w.writerow([f'{wp["x_m"]:.6f}', f'{wp["y_m"]:.6f}',
                         f'{wp["d_right"]:.6f}', f'{wp["d_left"]:.6f}'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="track name (also directory)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preset", choices=list(PRESETS), default="small")
    p.add_argument("--width", type=float, default=2.5,
                   help="track width [m]. 2026-05-27 default 2.5 (f map 비슷, mpc 안정).")
    p.add_argument("--vmax", type=float, default=6.0,
                   help="max reference speed [m/s] for waypoint vx_mps")
    p.add_argument("--ds", type=float, default=0.1, help="centerline resample step [m]")
    p.add_argument("--maps_dir", default=None,
                   help="output base directory (default: src/stack_master/maps)")
    p.add_argument("--a_lat_max", type=float, default=8.0,
                   help="lateral grip cap [m/s²] for forward-backward ref_v")
    p.add_argument("--a_long_max", type=float, default=4.0,
                   help="forward accel cap [m/s²] for forward-backward ref_v")
    p.add_argument("--a_brake_max", type=float, default=6.0,
                   help="braking decel cap [m/s²] (≥ a_long_max typically)")
    p.add_argument("--kappa_cap", type=float, default=0.55,
                   help="generator κ-threshold (default 0.55 allows hairpins; lib default 0.27)")
    args = p.parse_args()

    # Allow caller to override the track width by monkey-patching TRACK_WIDTH
    # before _create_track runs (the generator reads it at function scope).
    import random_track_generator.track_generator as _tg
    if args.width is not None:
        _tg.TRACK_WIDTH = float(args.width)
    # 곡률 cap 확장: 기본 1/3.75=0.267 (medium corner 까지) → 0.55 (hairpin 허용).
    # f map 같은 복잡한 트랙은 hairpin 이 필요. 사용자 요청 2026-05-26.
    _tg.CURVATURE_THRESHOLD = max(_tg.CURVATURE_THRESHOLD, args.kappa_cap)

    print(f"=== Generating random track '{args.name}' (preset={args.preset}, seed={args.seed}) ===")
    tr = gen_track_with_retry(args.preset, args.seed)
    cl_raw = tr.centerline
    width = float(tr.track_width)

    # Resample centerline
    cl_xy, _, total = resample_uniform(cl_raw, ds=args.ds)

    # ── Rotate centerline so index 0 = closest to (0,0) = car spawn pose ──
    # _create_track translates start_position to (0,0), but the spline's
    # parametric u=0 lands at sorted_vertices[0] which is unrelated to the
    # start position. Without this roll, the car spawns at s=L/2 on the
    # centerline → frenet projection garbage, lap counting broken,
    # vcmd stuck on ramp value.
    i0 = int(np.argmin(np.linalg.norm(cl_xy, axis=1)))
    if i0 != 0:
        cl_xy = np.roll(cl_xy, -i0, axis=0)
        print(f"  rolled centerline by {i0} pts so (0,0) is at s=0")

    psi, kappa = compute_psi_kappa(cl_xy, total)
    print(f"  centerline: {len(cl_xy)} pts, length={total:.2f}m, "
          f"|κ|max={np.abs(kappa).max():.3f}, "
          f"start=({cl_xy[0,0]:.2f},{cl_xy[0,1]:.2f})")

    # Build track polygon for occupancy.
    # Outer = centerline buffered +width/2, inner = -width/2.
    cl_line = LineString(np.vstack([cl_xy, cl_xy[0:1]]))
    outer = cl_line.buffer(width / 2, join_style=2)
    inner = cl_line.buffer(-width / 2, join_style=2)
    track_poly = outer.difference(inner)
    if track_poly.is_empty or not hasattr(track_poly, "exterior"):
        # Fallback: ring-shaped Polygon
        ring_outer = list(outer.exterior.coords)
        ring_inner = list(inner.exterior.coords) if hasattr(inner, "exterior") else []
        track_poly = Polygon(ring_outer, [ring_inner] if ring_inner else None)
    img, origin, resolution = render_occupancy(track_poly)

    gw = make_global_waypoints(cl_xy, psi, kappa, total,
                               width=width, vmax=args.vmax,
                               a_lat_max=args.a_lat_max,
                               a_long_max=args.a_long_max,
                               a_brake_max=args.a_brake_max)
    est_lap = gw["est_lap_time"]["data"]
    vx_arr = np.array([w["vx_mps"] for w in gw["global_traj_wpnts_iqp"]["wpnts"]])
    print(f"  est_lap_time={est_lap:.2f}s, "
          f"ref_v range=[{vx_arr.min():.2f}, {vx_arr.max():.2f}] m/s "
          f"(vmax cap {args.vmax}, a_lat={args.a_lat_max}, "
          f"a_long±=[{args.a_long_max}/{args.a_brake_max}])")

    maps_dir = Path(args.maps_dir) if args.maps_dir else (REPO_ROOT / "src/stack_master/maps")
    out_dir = maps_dir / args.name
    start_psi = float(psi[0])
    write_track(out_dir, args.name, img, origin, resolution, gw, width, est_lap,
                start_psi=start_psi)
    print(f"  wrote → {out_dir}")
    print(f"    {args.name}.png  {img.size[0]}×{img.size[1]} px @ {resolution} m/px")
    print(f"    origin = {origin}, start pose stheta={start_psi:.3f} rad")

    # Also drop the CSV bundle into nonlinear_mpc_acados/share/tracks/ so the
    # mpc_node CSV fallback works (it can't always rely on /centerline_waypoints
    # arriving in time, esp. with 0.5Hz republisher).
    csv_dir = REPO_ROOT / "src/nonlinear_mpc_acados/share/tracks"
    csv_sub = write_csv_track(csv_dir, args.name, cl_xy, psi, kappa, width, args.vmax,
                              total_length=total,
                              a_lat_max=args.a_lat_max,
                              a_long_max=args.a_long_max,
                              a_brake_max=args.a_brake_max)
    print(f"    csv → {csv_sub} (track_loader fallback)")


if __name__ == "__main__":
    main()
