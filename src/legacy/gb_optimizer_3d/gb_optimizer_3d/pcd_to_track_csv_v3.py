#!/usr/bin/env python3
"""Generate 3D track boundary CSV from rosbag poses + wall.pcd.

V3: Contour-based — extracts the wall PCD's inner contour directly,
then assigns left/right based on centerline. Gaps are filled via
spline interpolation.
"""

import os
import csv
import argparse
import numpy as np
import rosbag
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import cv2


RESOLUTION = 0.03  # grid resolution in meters


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


def wall_to_inner_contour(wall_pts_2d, close_kernel=5, min_contour_len=50):
    """Extract inner contours from wall point cloud (2D occupancy grid approach).

    Returns list of contour arrays, each (M, 2) in world coords.
    """
    xmin, ymin = wall_pts_2d.min(axis=0) - 0.5
    xmax, ymax = wall_pts_2d.max(axis=0) + 0.5

    nx = int((xmax - xmin) / RESOLUTION) + 1
    ny = int((ymax - ymin) / RESOLUTION) + 1
    grid = np.zeros((ny, nx), dtype=np.uint8)

    ix = ((wall_pts_2d[:, 0] - xmin) / RESOLUTION).astype(int)
    iy = ((wall_pts_2d[:, 1] - ymin) / RESOLUTION).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    grid[iy, ix] = 255

    print(f"  Grid: {nx}x{ny}, occupied: {(grid > 0).sum()}")

    # Morphological closing to fill small gaps in walls
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

    # Dilate slightly to connect nearby fragments
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.dilate(closed, dilate_k, iterations=2)
    closed = cv2.erode(closed, dilate_k, iterations=1)

    print(f"  After morphology: {(closed > 0).sum()} pixels")

    # Extract contours
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # Convert contours to world coordinates, filter small ones
    world_contours = []
    for i, c in enumerate(contours):
        c = c.squeeze()
        if c.ndim < 2 or len(c) < min_contour_len:
            continue
        wc = np.zeros((len(c), 2))
        wc[:, 0] = c[:, 0] * RESOLUTION + xmin
        wc[:, 1] = c[:, 1] * RESOLUTION + ymin
        world_contours.append(wc)

    print(f"  Found {len(world_contours)} contours (from {len(contours)} total)")
    return world_contours, closed, (xmin, ymin, nx, ny)


def assign_contour_to_boundary(contour_pts, centerline, tangent_yaw):
    """For each contour point, determine if it's left or right of centerline.

    Returns:
        left_pts: (K, 2) array of left boundary points (ordered by s)
        right_pts: (K, 2) array of right boundary points (ordered by s)
        left_s: corresponding s indices
        right_s: corresponding s indices
    """
    cl_tree = cKDTree(centerline[:, :2])

    left_pts = []
    left_s = []
    right_pts = []
    right_s = []

    for pt in contour_pts:
        _, ci = cl_tree.query(pt)
        yaw = tangent_yaw[ci]
        # Left normal direction
        lnx = -np.sin(yaw)
        lny = np.cos(yaw)
        dx = pt[0] - centerline[ci, 0]
        dy = pt[1] - centerline[ci, 1]
        cross = dx * lny - dy * lnx  # positive = left side

        if cross > 0:
            left_pts.append(pt)
            left_s.append(ci)
        else:
            right_pts.append(pt)
            right_s.append(ci)

    return (np.array(left_pts) if left_pts else np.empty((0, 2)),
            np.array(left_s) if left_s else np.empty(0, dtype=int),
            np.array(right_pts) if right_pts else np.empty((0, 2)),
            np.array(right_s) if right_s else np.empty(0, dtype=int))


def contour_pts_to_boundary(pts, s_indices, centerline, n_centerline):
    """Convert scattered contour points to a per-centerline-point boundary.

    For each centerline point, find the closest contour point assigned to it.
    Returns (n_centerline, 2) boundary array and found mask.
    """
    bound = np.zeros((n_centerline, 2))
    found = np.zeros(n_centerline, dtype=bool)

    if len(pts) == 0:
        return bound, found

    # For each centerline index, collect assigned contour points and pick closest
    for ci in range(n_centerline):
        mask = s_indices == ci
        if not mask.any():
            continue
        nearby = pts[mask]
        dists = np.linalg.norm(nearby - centerline[ci, :2], axis=1)
        closest = np.argmin(dists)
        bound[ci] = nearby[closest]
        found[ci] = True

    return bound, found


def interpolate_boundary(bound, found, n):
    """Fill missing boundary points via circular interpolation."""
    if found.all():
        return bound
    valid_idx = np.where(found)[0]
    missing_idx = np.where(~found)[0]
    if len(valid_idx) == 0:
        return bound
    for dim in range(2):
        bound[missing_idx, dim] = np.interp(
            missing_idx, valid_idx, bound[valid_idx, dim], period=n)
    return bound


def main():
    parser = argparse.ArgumentParser(description="V3: Contour-based track boundary")
    parser.add_argument("--bag", required=True, help="Input bag file")
    parser.add_argument("--wall", default="pcd/wall.pcd")
    parser.add_argument("--pose-topic", default="/glim_ros/pose_corrected")
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--close-kernel", type=int, default=7,
                        help="Morphological closing kernel size [pixels]")
    parser.add_argument("--min-dist", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw_track_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(output_dir, "localization_bridge_2_bounds_3d.csv")

    # 1. Load wall
    walls = np.asarray(o3d.io.read_point_cloud(args.wall).points)
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

    poses = [raw_poses[0]]
    for p in raw_poses[1:]:
        if np.linalg.norm(p[:3] - poses[-1][:3]) >= args.min_dist:
            poses.append(p)
    poses = np.array(poses)

    dists = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    half = len(poses) // 2
    min_idx = half + np.argmin(dists[half:])
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

    # 4. Extract wall inner contours
    print("Extracting wall contours...")
    contours, grid_img, (xmin, ymin, grid_nx, grid_ny) = wall_to_inner_contour(
        walls[:, :2], close_kernel=args.close_kernel)

    # 5. Assign all contour points to left/right
    print("Assigning contour points to boundaries...")
    all_left_pts = []
    all_left_s = []
    all_right_pts = []
    all_right_s = []

    for contour in contours:
        lp, ls, rp, rs = assign_contour_to_boundary(contour, poses, tangent_yaw)
        if len(lp) > 0:
            all_left_pts.append(lp)
            all_left_s.append(ls)
        if len(rp) > 0:
            all_right_pts.append(rp)
            all_right_s.append(rs)

    all_left_pts = np.vstack(all_left_pts) if all_left_pts else np.empty((0, 2))
    all_left_s = np.concatenate(all_left_s) if all_left_s else np.empty(0, dtype=int)
    all_right_pts = np.vstack(all_right_pts) if all_right_pts else np.empty((0, 2))
    all_right_s = np.concatenate(all_right_s) if all_right_s else np.empty(0, dtype=int)

    print(f"  Left contour points: {len(all_left_pts)}, Right: {len(all_right_pts)}")

    # 6. Convert to per-centerline boundary
    left_bound, left_found = contour_pts_to_boundary(all_left_pts, all_left_s, poses, n)
    right_bound, right_found = contour_pts_to_boundary(all_right_pts, all_right_s, poses, n)

    print(f"  Found: left={left_found.sum()}/{n}, right={right_found.sum()}/{n}")

    # 7. Outlier filter
    for arr, found, label in [(left_bound, left_found, "left"),
                               (right_bound, right_found, "right")]:
        widths = np.linalg.norm(arr - poses[:, :2], axis=1)
        valid_w = widths[found]
        if len(valid_w) == 0:
            continue
        median_w = np.median(valid_w)
        max_w = max(median_w * 2.5, 1.5)
        outlier = found & (widths > max_w)
        if outlier.any():
            print(f"  [{label}] Filtered {outlier.sum()} outliers (>{max_w:.3f}m)")
            found[outlier] = False

    # 8. Interpolate gaps
    left_bound = interpolate_boundary(left_bound, left_found, n)
    right_bound = interpolate_boundary(right_bound, right_found, n)

    print(f"  After interpolation: left gaps={(~left_found).sum()}, right gaps={(~right_found).sum()}")

    # 9. Smooth
    if args.smooth_window > 0:
        left_bound = smooth_circular(left_bound, args.smooth_window)
        right_bound = smooth_circular(right_bound, args.smooth_window)

    # 10. Add z from poses
    left_bound_3d = np.column_stack([left_bound, poses[:, 2]])
    right_bound_3d = np.column_stack([right_bound, poses[:, 2]])

    # 11. Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['right_bound_x', 'right_bound_y', 'right_bound_z',
                         'left_bound_x', 'left_bound_y', 'left_bound_z'])
        for i in range(n):
            writer.writerow([
                f"{right_bound_3d[i,0]:.6f}", f"{right_bound_3d[i,1]:.6f}", f"{right_bound_3d[i,2]:.6f}",
                f"{left_bound_3d[i,0]:.6f}", f"{left_bound_3d[i,1]:.6f}", f"{left_bound_3d[i,2]:.6f}"
            ])

    wl = np.linalg.norm(left_bound - poses[:, :2], axis=1)
    wr = np.linalg.norm(right_bound - poses[:, :2], axis=1)
    print(f"Left:  {wl.min():.3f} ~ {wl.max():.3f}m (mean {wl.mean():.3f})")
    print(f"Right: {wr.min():.3f} ~ {wr.max():.3f}m (mean {wr.mean():.3f})")
    print(f"Saved {n} points to {output_path}")

    # 12. Debug plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: grid + contours
    ax = axes[0]
    ax.imshow(grid_img, origin='lower', cmap='gray', alpha=0.5,
              extent=[xmin, xmin + grid_nx * RESOLUTION, ymin, ymin + grid_ny * RESOLUTION])
    for c in contours:
        ax.plot(c[:, 0], c[:, 1], '-', linewidth=0.5, alpha=0.7)
    ax.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=1, label='centerline')
    ax.set_aspect('equal')
    ax.set_title('Wall grid + contours')
    ax.legend()

    # Right: final boundary
    ax = axes[1]
    ax.scatter(walls[:, 0], walls[:, 1], s=0.1, c='gray', alpha=0.3, label='wall pcd')
    ax.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=1, label='centerline')
    ax.plot(left_bound[:, 0], left_bound[:, 1], 'g-', linewidth=1.5, label='left bound')
    ax.plot(right_bound[:, 0], right_bound[:, 1], 'r-', linewidth=1.5, label='right bound')
    # Mark interpolated regions
    ax.scatter(left_bound[~left_found, 0], left_bound[~left_found, 1],
               c='lime', s=10, zorder=5, label='left interp')
    ax.scatter(right_bound[~right_found, 0], right_bound[~right_found, 1],
               c='orange', s=10, zorder=5, label='right interp')
    ax.set_aspect('equal')
    ax.set_title('V3: Contour-based boundary')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
