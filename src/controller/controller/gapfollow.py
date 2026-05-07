import math
import heapq
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped

from controller.estop import EStop


class GapFollowNode(Node):

    # ── 파라미터 ──────────────────────────────────────────────
    GRID_RESOLUTION = 0.10        # m/cell
    GRID_WIDTH = 100              # cells, 전방 x
    GRID_HEIGHT = 200             # cells, 측면 y
    GRID_REAR = 10                # cells, 후방 x
    MAX_SCAN_RANGE = 10.0         # m
    SCAN_FOV = math.pi / 2        # rad (정면 ±90도)
    RAY_COUNT = 360               # 레이캐스팅 빔 수 (정면 180도)
    HARD_INFLATION = 0.1          # m, scan point 주변 100으로 채우는 반경
    SOFT_INFLATION = 0.5          # m, 정규분포 cost 확장 반경
    LASER_OFFSET_X = 0.0          # m, laser→base_link x 오프셋
    WHEELBASE = 0.33              # m
    MAX_STEER = 0.4               # rad
    SPEED = 3.0                   # m/s
    SPEED_MIN = 1.5               # m/s (최대 조향 시)
    START_CLEAR_EXTRA = 2         # inflation 외 추가 클리어 셀 수
    ASTAR_MAX_ITER = 15000        # A* 최대 탐색 노드 수
    INFLATION_COST_SCALE = 3.0    # inflate 셀 cost 배율 (1~99 → 1x ~ Nx)
    UNKNOWN_COST = 3.0            # unknown(-1) 셀 통과 시 추가 cost 배율
    PP_LOOKAHEAD = 0.5            # m
    MARKER_SPHERE_SIZE = 0.3      # m
    MARKER_LINE_WIDTH = 0.05      # m

    def __init__(self):
        super().__init__('gap_follow')

        self.estop = EStop(self)
        self.total_w = self.GRID_WIDTH + self.GRID_REAR

        # Hard inflation kernel (binary circle)
        hard_cells = int(math.ceil(self.HARD_INFLATION / self.GRID_RESOLUTION))
        hy, hx = np.mgrid[-hard_cells:hard_cells+1, -hard_cells:hard_cells+1]
        self.hard_kernel = ((hx**2 + hy**2) <= hard_cells**2).astype(np.uint8)
        self.hard_cells = hard_cells

        # Soft inflation kernel (gaussian, 0~99)
        sigma_cells = self.SOFT_INFLATION / self.GRID_RESOLUTION / 2.0
        soft_cells = int(math.ceil(3.0 * sigma_cells))
        sy, sx = np.mgrid[-soft_cells:soft_cells+1, -soft_cells:soft_cells+1]
        gauss = np.exp(-(sx**2 + sy**2) / (2.0 * sigma_cells**2))
        self.soft_kernel = (gauss * 99).astype(np.int8)
        self.soft_cells = soft_cells

        # Raycasting 각도 배열 (정면 180도)
        self.ray_angles = np.linspace(-self.SCAN_FOV, self.SCAN_FOV, self.RAY_COUNT)
        self.ray_max_steps = int(self.MAX_SCAN_RANGE / self.GRID_RESOLUTION)

        self.odom = None

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/vesc/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/grid_map', 10)
        self.path_pub = self.create_publisher(Path, '/astar_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/pp_markers', 10)

        self.get_logger().info('GapFollowNode ready')

    def _scan_cb(self, msg):
        self.scan = msg
        if self.odom is None:
            return

        steer, speed = self._compute()
        drive = AckermannDriveStamped()
        drive.header.stamp = msg.header.stamp
        drive.header.frame_id = 'base_link'
        drive.drive.steering_angle = steer
        drive.drive.speed = speed
        self.drive_pub.publish(drive)

    def _odom_cb(self, msg):
        self.odom = msg

    def _compute(self):
        scan_stamp = self.scan.header.stamp

        grid = self._build_grid()
        self._publish_grid(grid, scan_stamp)

        start_cell = self._world_to_grid(-self.LASER_OFFSET_X, 0.0)
        sr, sc = start_cell
        clear_r = self.soft_cells + self.START_CLEAR_EXTRA
        r0 = max(sr - clear_r, 0)
        r1 = min(sr + clear_r + 1, self.GRID_HEIGHT)
        c0 = max(sc - clear_r, 0)
        c1 = min(sc + clear_r + 1, self.total_w)
        grid[r0:r1, c0:c1] = 0

        goal_cell = self._farthest_free(grid, sr, sc)
        if goal_cell is None:
            return 0.0, self.SPEED

        path_cells = self._astar(grid, start_cell, goal_cell)
        if not path_cells or len(path_cells) < 2:
            return 0.0, self.SPEED

        path_world = [self._grid_to_world(r, c) for r, c in path_cells]
        self._publish_path(path_world, scan_stamp)

        steer, lookahead_pt = self._pure_pursuit(path_world)
        steer = float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER))
        self._publish_markers(lookahead_pt, scan_stamp)

        # 조향각에 비례해서 감속 (직선=SPEED, 최대조향=SPEED_MIN)
        ratio = abs(steer) / self.MAX_STEER
        speed = self.SPEED - ratio * (self.SPEED - self.SPEED_MIN)
        return steer, speed

    # ── Occupancy Grid (4-step pipeline) ──────────────────────

    def _build_grid(self):
        H, W = self.GRID_HEIGHT, self.total_w
        cy = H // 2
        res = self.GRID_RESOLUTION

        # Step 1: unknown grid
        grid = np.full((H, W), -1, dtype=np.int8)

        # Step 2: scan points → occupied + hard inflation
        ranges = np.array(self.scan.ranges)
        angles = self.scan.angle_min + np.arange(len(ranges)) * self.scan.angle_increment
        valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges <= self.MAX_SCAN_RANGE)
        ranges = ranges[valid]
        angles = angles[valid]
        front = np.abs(angles) < self.SCAN_FOV
        ranges = ranges[front]
        angles = angles[front]

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        cols_pt = np.round(xs / res).astype(int) + self.GRID_REAR
        rows_pt = np.round(-ys / res).astype(int) + cy
        in_b = (rows_pt >= 0) & (rows_pt < H) & (cols_pt >= 0) & (cols_pt < W)

        # 끝점 마킹 + hard inflation (100)
        pad_h = self.hard_cells
        for r, c in zip(rows_pt[in_b], cols_pt[in_b]):
            r0 = max(r - pad_h, 0)
            r1 = min(r + pad_h + 1, H)
            c0 = max(c - pad_h, 0)
            c1 = min(c + pad_h + 1, W)
            kr0 = pad_h - (r - r0)
            kr1 = pad_h + (r1 - r)
            kc0 = pad_h - (c - c0)
            kc1 = pad_h + (c1 - c)
            k = self.hard_kernel[kr0:kr1, kc0:kc1]
            region = grid[r0:r1, c0:c1]
            region[k == 1] = 100

        # Step 3: 로봇 기준 정면 180도 레이캐스팅 (DDA) → 100 만날때까지 free(0)
        origin_r, origin_c = cy, self.GRID_REAR
        for ang in self.ray_angles:
            dx = math.cos(ang)
            dy = -math.sin(ang)  # row는 y 반전
            # DDA: 1셀씩 정밀 이동
            if abs(dx) > abs(dy):
                step = abs(dx)
            else:
                step = abs(dy)
            if step < 1e-9:
                continue
            dr = dy / step
            dc = dx / step
            r, c = float(origin_r), float(origin_c)
            for _ in range(self.ray_max_steps):
                r += dr
                c += dc
                ri, ci = int(round(r)), int(round(c))
                if ri < 0 or ri >= H or ci < 0 or ci >= W:
                    break
                if grid[ri, ci] == 100:
                    break
                grid[ri, ci] = 0  # free

        # Step 4: soft inflation — free(0) 영역만 정규분포 cost 확장
        pad_s = self.soft_cells
        occ_rows, occ_cols = np.where(grid == 100)
        for r, c in zip(occ_rows, occ_cols):
            r0 = max(r - pad_s, 0)
            r1 = min(r + pad_s + 1, H)
            c0 = max(c - pad_s, 0)
            c1 = min(c + pad_s + 1, W)
            kr0 = pad_s - (r - r0)
            kr1 = pad_s + (r1 - r)
            kc0 = pad_s - (c - c0)
            kc1 = pad_s + (c1 - c)
            k = self.soft_kernel[kr0:kr1, kc0:kc1]
            region = grid[r0:r1, c0:c1]
            can = (region >= 0) & (region < 100)
            region[can] = np.maximum(region[can], k[can])

        return grid

    # ── A* ─────────────────────────────────────────────────────

    def _astar(self, grid, start, goal):
        sr, sc = start
        gr, gc = goal

        if grid[gr, gc] == 100:
            gr, gc = self._nearest_free(grid, gr, gc)
            if gr is None:
                return []

        open_set = [(0.0, sr, sc)]
        came_from = {}
        g_score = {(sr, sc): 0.0}
        closed = set()
        neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
        sqrt2 = math.sqrt(2)

        iterations = 0
        while open_set:
            if iterations >= self.ASTAR_MAX_ITER:
                break
            iterations += 1
            _, cr, cc = heapq.heappop(open_set)

            if (cr, cc) in closed:
                continue
            closed.add((cr, cc))

            if cr == gr and cc == gc:
                path = [(cr, cc)]
                while (cr, cc) in came_from:
                    cr, cc = came_from[(cr, cc)]
                    path.append((cr, cc))
                path.reverse()
                return path

            for dr, dc in neighbors_8:
                nr, nc = cr + dr, cc + dc
                if nr < 0 or nr >= self.GRID_HEIGHT or nc < 0 or nc >= self.total_w:
                    continue
                if (nr, nc) in closed:
                    continue
                cell = int(grid[nr, nc])
                if cell == 100:
                    continue

                cost = sqrt2 if (dr != 0 and dc != 0) else 1.0
                if cell == -1:
                    cost *= self.UNKNOWN_COST
                elif cell > 0:
                    cost *= 1.0 + (cell / 99.0) * (self.INFLATION_COST_SCALE - 1.0)
                tg = g_score[(cr, cc)] + cost

                if tg < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = tg
                    came_from[(nr, nc)] = (cr, cc)
                    h = math.hypot(nr - gr, nc - gc)
                    heapq.heappush(open_set, (tg + h, nr, nc))

        return []

    def _nearest_free(self, grid, r, c, max_iter=1000):
        q = deque([(r, c)])
        visited = {(r, c)}
        count = 0
        while q and count < max_iter:
            count += 1
            cr, cc = q.popleft()
            if grid[cr, cc] != 100:
                return cr, cc
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.total_w:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
        return None, None

    def _farthest_free(self, grid, sr, sc):
        free_rows, free_cols = np.where((grid >= 0) & (grid < 100))
        if len(free_rows) == 0:
            return None
        dist_sq = (free_rows - sr) ** 2 + (free_cols - sc) ** 2
        idx = np.argmax(dist_sq)
        return int(free_rows[idx]), int(free_cols[idx])

    # ── Pure Pursuit ───────────────────────────────────────────

    def _pure_pursuit(self, path):
        lookahead_pt = None
        for x, y in path:
            if math.hypot(x, y) >= self.PP_LOOKAHEAD:
                lookahead_pt = (x, y)
                break
        if lookahead_pt is None:
            lookahead_pt = path[-1]

        tx, ty = lookahead_pt
        ld = math.hypot(tx, ty)
        if ld < 1e-6:
            return 0.0, lookahead_pt

        alpha = math.atan2(ty, tx)
        steer = math.atan2(2.0 * self.WHEELBASE * math.sin(alpha), ld)
        return steer, lookahead_pt

    # ── Coordinate transforms ─────────────────────────────────

    def _world_to_grid(self, x, y):
        cy = self.GRID_HEIGHT // 2
        col = int(round(x / self.GRID_RESOLUTION)) + self.GRID_REAR
        row = int(round(-y / self.GRID_RESOLUTION)) + cy
        return row, col

    def _grid_to_world(self, row, col):
        cy = self.GRID_HEIGHT // 2
        x = (col - self.GRID_REAR) * self.GRID_RESOLUTION
        y = -(row - cy) * self.GRID_RESOLUTION
        return x, y

    # ── Publishers ─────────────────────────────────────────────

    def _publish_grid(self, grid, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = 'laser'
        msg.info.resolution = float(self.GRID_RESOLUTION)
        msg.info.width = self.total_w
        msg.info.height = self.GRID_HEIGHT
        half_y = self.GRID_HEIGHT * self.GRID_RESOLUTION / 2.0
        msg.info.origin.position.x = -self.GRID_REAR * self.GRID_RESOLUTION
        msg.info.origin.position.y = -half_y
        msg.info.origin.position.z = 0.0
        flipped = np.flipud(grid)
        msg.data = flipped.flatten().tolist()
        self.grid_pub.publish(msg)

    def _publish_path(self, path_world, stamp):
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = 'laser'
        for x, y in path_world:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def _publish_markers(self, lookahead_pt, stamp):
        ma = MarkerArray()

        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = 'laser'
        m.ns = 'pp_lookahead'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(lookahead_pt[0])
        m.pose.position.y = float(lookahead_pt[1])
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = self.MARKER_SPHERE_SIZE
        m.scale.y = self.MARKER_SPHERE_SIZE
        m.scale.z = self.MARKER_SPHERE_SIZE
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        ma.markers.append(m)

        line = Marker()
        line.header.stamp = stamp
        line.header.frame_id = 'laser'
        line.ns = 'pp_line'
        line.id = 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = self.MARKER_LINE_WIDTH
        line.color.r = 1.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.color.a = 1.0
        line.points.append(Point(x=0.0, y=0.0, z=0.0))
        line.points.append(Point(
            x=float(lookahead_pt[0]),
            y=float(lookahead_pt[1]),
            z=0.0))
        ma.markers.append(line)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = GapFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
