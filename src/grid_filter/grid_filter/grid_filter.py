"""GridFilter — OccupancyGrid 기반 침식 (erosion) 필터 헬퍼 (ROS2 포팅).

원본 ROS1: f110_utils/libs/grid_filter/src/grid_filter/grid_filter.py.
ROS2 변경:
- rospy → rclpy. 단 GridFilter 는 Node 가 아니라 helper class 이므로 외부에서 node 를
  주입받아 그 node 의 create_subscription / get_logger 을 사용한다.
- 호출 패턴 (ROS1):  GridFilter(map_topic="/map", debug=False)  ← 자체 rospy.Subscriber
- 호출 패턴 (ROS2):  GridFilter(node=self, map_topic="/map", debug=False)
  (node=None 이면 subscribe 안 함, load_from_file 로만 사용 가능)
"""
from __future__ import annotations

import cv2
import numpy as np
import yaml

from nav_msgs.msg import OccupancyGrid


class GridFilter:
    def __init__(self, node=None, map_topic=None, debug=False):
        self.resolution = None  # m/pixel
        self.origin = None  # (x, y)
        self.map_data = None
        self.image = None  # OccupancyGrid → OpenCV image
        self.eroded_image = None
        self.kernel_size = 3
        self.debug = debug
        self.map_topic = map_topic
        self._node = node

        if self.map_topic and self._node is not None:
            self.subscribe_to_map(self.map_topic)
        elif self.map_topic and self._node is None:
            # 사용자 의도 없이 자동 구독 시도했지만 node 가 없음 → 경고만
            print("[GridFilter] map_topic 가 주어졌지만 node=None — 자동 구독 안 함. "
                  "load_from_file 로 직접 로드하거나 node 를 주입할 것.")

    # ---- ROS subscribe ----

    def subscribe_to_map(self, map_topic):
        if self._node is None:
            return
        self._node.get_logger().info(f"[GridFilter] subscribing to {map_topic}")
        self._node.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, 10
        )

    def map_callback(self, msg: OccupancyGrid):
        if self.image is not None:
            return
        if self._node is not None:
            self._node.get_logger().warn("[GridFilter] received map data")

        self.resolution = msg.info.resolution
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        width, height = msg.info.width, msg.info.height
        image = np.array(msg.data, dtype=np.int8).reshape((height, width))
        # 100 → obstacle, 그 외 → free
        self.image = np.where(image == 100, 0, 255).astype(np.uint8)

        if self.debug:
            cv2.imshow("Original Map with Points", self.image)
            cv2.waitKey(0)

        self.update_image()
        if self._node is not None:
            self._node.get_logger().warn("[GridFilter] map image initialized")

    # ---- 파일 로드 ----

    def load_from_file(self, png_path: str, yaml_path: str) -> bool:
        """PNG + YAML 으로 직접 로드 (ROS subscribe 없이)."""
        try:
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
            self.resolution = yaml_data["resolution"]
            self.origin = (yaml_data["origin"][0], yaml_data["origin"][1])

            loaded_image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if loaded_image is None:
                if self._node is not None:
                    self._node.get_logger().error(f"[GridFilter] failed to load image: {png_path}")
                return False
            # PNG (top-down) → ROS OccupancyGrid (bottom-up). 좌표 일치하도록 Y 반전
            self.image = cv2.flip(loaded_image, 0)

            if self.debug:
                cv2.imshow("Loaded Map from File", self.image)
                cv2.waitKey(0)
            self.update_image()
            if self._node is not None:
                self._node.get_logger().info(f"[GridFilter] map loaded: {png_path}")
            return True
        except Exception as e:  # noqa: BLE001
            if self._node is not None:
                self._node.get_logger().error(f"[GridFilter] load failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ---- erosion ----

    def set_erosion_kernel_size(self, size: int):
        self.kernel_size = size
        self.update_image()

    def update_image(self):
        if self.image is None:
            if self._node is not None:
                self._node.get_logger().warn("[GridFilter] map image not initialized.")
            return
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        self.eroded_image = cv2.erode(self.image, kernel)
        if self.debug:
            cv2.imshow("Eroded Map", self.eroded_image)
            cv2.waitKey(0)

    # ---- 좌표 ----

    def world_to_pixel(self, x, y):
        px = int((x - self.origin[0]) / self.resolution)
        py = int((y - self.origin[1]) / self.resolution)
        return px, py

    def is_point_inside(self, x, y) -> bool:
        if self.eroded_image is None:
            return False
        px, py = self.world_to_pixel(x, y)
        if px < 0 or py < 0 or px >= self.eroded_image.shape[1] or py >= self.eroded_image.shape[0]:
            return False
        return bool(self.eroded_image[py, px] == 255)
