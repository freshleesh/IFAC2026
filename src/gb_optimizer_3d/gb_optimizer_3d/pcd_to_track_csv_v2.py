#!/usr/bin/env python3
"""Generate 3D track boundary CSV from rosbag poses + wall.pcd.

V2: Density-based ray casting — instead of picking the single closest point,
bins points along each ray by distance and picks the densest cluster's median.
This is robust to SLAM drift ghost points and dynamic object remnants.
"""

import os
import csv
import argparse
import numpy as np
import rosbag
import open3d as o3d
from scipy.spatial import cKDTree


def quat_to_yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def smooth_circular(arr, window):
    n = len(arr)
    if window <= 0:
        return arr
    padded = np.concatenate([arr[-window:], arr, arr[:window]], axis=0)
    smoothed = np.zeros_like(arr)
    for i in range(n):
        smoothed[i] = np.mean(padded[i:i + 2 * window + 1], axis=0)
    return smoothed


def find_boundary_ray_v2(cx, cy, yaw, wall_tree, walls, direction,
                         ray_width=0.3, max_search=2.0, bin_size=0.05, min_pts=3):
    """Density-based ray casting.

    1. Collect all wall points within the ray strip
    2. Bin them by distance along the ray
    3. Find the bin with the most points (= actual wall)
    4. Return median position of points in that bin

    Returns (x, y) of wall point or None.
    """
    lnx = -np.sin(yaw) * direction
    lny = np.cos(yaw) * direction
    tx = np.cos(yaw)
    ty = np.sin(yaw)

    idx_list = wall_tree.query_ball_point([cx, cy], max_search)
    if not idx_list:
        return None

    nearby = walls[idx_list]
    dx = nearby[:, 0] - cx
    dy = nearby[:, 1] - cy

    ray_proj = dx * lnx + dy * lny   # distance along ray
    t_proj = dx * tx + dy * ty        # distance along tangent

    mask = (np.abs(t_proj) < ray_width) & (ray_proj > 0.05)

    if not mask.any():
        return None

    candidates = nearby[mask]
    ray_dists = ray_proj[mask]

    # If very few points, fall back to closest
    if len(ray_dists) < min_pts:
        closest = np.argmin(ray_dists)
        return candidates[closest, :2]

    # Bin by distance along ray
    d_min, d_max = ray_dists.min(), ray_dists.max()
    if d_max - d_min < bin_size:
        # All points in one bin
        return np.median(candidates[:, :2], axis=0)

    n_bins = max(1, int((d_max - d_min) / bin_size))
    bin_edges = np.linspace(d_min, d_max + 1e-6, n_bins + 1)
    bin_indices = np.digitize(ray_dists, bin_edges) - 1

    # Find densest bin
    counts = np.bincount(bin_indices, minlength=n_bins)
    best_bin = np.argmax(counts)

    # Points in the densest bin
    in_best = bin_indices == best_bin
    if in_best.sum() == 0:
        return None

    return np.median(candidates[in_best, :2], axis=0)


def main():
    parser = argparse.ArgumentParser(description="V2: Density-based track boundary from bag poses + wall PCD")
    parser.add_argument("--bag", required=True, help="Input bag file")
    parser.add_argument("--wall", default="pcd/wall.pcd")
    parser.add_argument("--pose-topic", default="/glim_ros/pose_corrected")
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--ray-width", type=float, default=0.3)
    parser.add_argument("--max-search", type=float, default=2.0)
    parser.add_argument("--bin-size", type=float, default=0.05,
                        help="Histogram bin size along ray [m]")
    parser.add_argument("--min-pts", type=int, default=3,
                        help="Minimum points for density mode (else fallback to closest)")
    parser.add_argument("--min-dist", type=float, default=0.05,
                        help="Minimum distance between centerline points [m]")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw_track_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(output_dir, "localization_bridge_2_bounds_3d.csv")

    # 1. Load wall
    walls = np.asarray(o3d.io.read_point_cloud(args.wall).points)
    wall_tree = cKDTree(walls[:, :2])
    print(f"Wall: {len(walls)} points")

    # 2. Extract poses (1 loop)
    print("Extracting poses...")
    bag = rosbag.Bag(args.bag)
    raw_poses = []
    for _, msg, _ in bag.read_messages(topics=[args.pose_topic]):
        p = msg.pose.position
        yaw = quat_to_yaw(msg.pose.orientation)
        raw_poses.append([p.x, p.y, p.z, yaw])
    bag.close()
    raw_poses = np.array(raw_poses)
    print(f"  Raw poses: {len(raw_poses)}")

    # Downsample
    poses = [raw_poses[0]]
    for p in raw_poses[1:]:
        if np.linalg.norm(p[:3] - poses[-1][:3]) >= args.min_dist:
            poses.append(p)
    poses = np.array(poses)

    # Close loop
    dists = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    half = len(poses) // 2
    min_idx = half + np.argmin(dists[half:])
    print(f"  Loop closure at idx {min_idx}, dist={dists[min_idx]:.4f}m")
    poses = poses[:min_idx + 1]
    poses[-1] = poses[0].copy()
    n = len(poses)
    print(f"  Centerline: {n} points")

    # 3. Tangent yaw
    tangent_yaw = np.zeros(n)
    for i in range(n):
        dx = poses[(i + 1) % n, 0] - poses[(i - 1) % n, 0]
        dy = poses[(i + 1) % n, 1] - poses[(i - 1) % n, 1]
        tangent_yaw[i] = np.arctan2(dy, dx)

    # 4. Density-based ray cast to wall
    print("Finding boundaries (density-based)...")
    left_bound = np.zeros((n, 3))
    right_bound = np.zeros((n, 3))
    left_found = np.zeros(n, dtype=bool)
    right_found = np.zeros(n, dtype=bool)

    for i in range(n):
        cx, cy, cz = poses[i, :3]
        yaw = tangent_yaw[i]

        lpt = find_boundary_ray_v2(cx, cy, yaw, wall_tree, walls, +1,
                                   args.ray_width, args.max_search, args.bin_size, args.min_pts)
        rpt = find_boundary_ray_v2(cx, cy, yaw, wall_tree, walls, -1,
                                   args.ray_width, args.max_search, args.bin_size, args.min_pts)

        if lpt is not None:
            left_bound[i] = [lpt[0], lpt[1], cz]
            left_found[i] = True
        if rpt is not None:
            right_bound[i] = [rpt[0], rpt[1], cz]
            right_found[i] = True

    # 5. Filter outlier widths
    for arr, found, label in [(left_bound, left_found, "left"),
                               (right_bound, right_found, "right")]:
        widths = np.linalg.norm(arr[:, :2] - poses[:, :2], axis=1)
        valid_widths = widths[found]
        if len(valid_widths) == 0:
            continue
        median_w = np.median(valid_widths)
        max_w = max(median_w * 2.5, 1.5)
        outlier = found & (widths > max_w)
        if outlier.any():
            print(f"  [{label}] Filtered {outlier.sum()} outliers (>{max_w:.3f}m)")
            found[outlier] = False

    print(f"  Missing: left={(~left_found).sum()}, right={(~right_found).sum()}")

    # 6. Interpolate missing
    for arr, found in [(left_bound, left_found), (right_bound, right_found)]:
        if found.all():
            continue
        valid_idx = np.where(found)[0]
        missing_idx = np.where(~found)[0]
        if len(valid_idx) == 0:
            continue
        for dim in range(2):
            arr[missing_idx, dim] = np.interp(
                missing_idx, valid_idx, arr[valid_idx, dim], period=n)

    if args.smooth_window > 0:
        left_bound = smooth_circular(left_bound, args.smooth_window)
        right_bound = smooth_circular(right_bound, args.smooth_window)

    # 7. Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['right_bound_x', 'right_bound_y', 'right_bound_z',
                         'left_bound_x', 'left_bound_y', 'left_bound_z'])
        for i in range(n):
            writer.writerow([
                f"{right_bound[i,0]:.6f}", f"{right_bound[i,1]:.6f}", f"{right_bound[i,2]:.6f}",
                f"{left_bound[i,0]:.6f}", f"{left_bound[i,1]:.6f}", f"{left_bound[i,2]:.6f}"
            ])

    wl = np.linalg.norm(left_bound[:, :2] - poses[:, :2], axis=1)
    wr = np.linalg.norm(right_bound[:, :2] - poses[:, :2], axis=1)
    print(f"Left:  {wl.min():.3f} ~ {wl.max():.3f}m (mean {wl.mean():.3f})")
    print(f"Right: {wr.min():.3f} ~ {wr.max():.3f}m (mean {wr.mean():.3f})")
    print(f"Saved {n} points to {output_path}")

    # 8. Debug plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(walls[:, 0], walls[:, 1], s=0.1, c='gray', alpha=0.3, label='wall pcd')
    ax.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=1, label='centerline (odom)')
    ax.plot(left_bound[:, 0], left_bound[:, 1], 'g-', linewidth=1, label='left bound')
    ax.plot(right_bound[:, 0], right_bound[:, 1], 'r-', linewidth=1, label='right bound')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('V2: Density-based boundary detection')
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
