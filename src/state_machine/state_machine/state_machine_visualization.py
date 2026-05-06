"""Visualization mixin — state_machine 노드의 RViz 마커 발행 메서드 모음.

이 mixin이 의존하는 self attribute (StateMachine 본체 `_init_state_attributes` 에서 초기화):
    - `cur_gb_wpnts.list`             — visualization anchor 계산 + not_ready 마커 위치 평균
    - `first_visualization`           — anchor 1회만 계산하기 위한 플래그
    - `x_viz`, `y_viz`                — 캐시된 anchor 좌표
    - `cur_volt`                      — battery voltage 텍스트
    - `local_wpnts_src`               — wpnts 출처 텍스트
    - 다음 publishers (StateMachine `_setup_ros_publishers` 에서 생성):
        * `vis_loc_wpnt_pub`, `vis_loc_vel_pub`, `loc_wpnt_pub`
        * `state_mrk`, `state_wpnts_src_marker`, `emergency_pub`
"""
import numpy as np
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import Marker, MarkerArray


# state name -> sphere marker (r, g, b) — visualize_state 시각화용
STATE_COLORS = {
    "GB_TRACK":     (0.0, 0.0, 1.0),  # Blue
    "OVERTAKE":     (1.0, 0.0, 0.0),  # Red
    "TRAILING":     (1.0, 1.0, 0.0),  # Yellow
    "ATTACK":       (1.0, 0.0, 1.0),  # Magenta
    "FTGONLY":      (1.0, 1.0, 1.0),  # White
    "RECOVERY":     (0.0, 1.0, 0.0),  # Green
    "SMART_STATIC": (0.0, 1.0, 1.0),  # Cyan
}
_STATE_COLOR_DEFAULT = (1.0, 1.0, 1.0)  # 알 수 없는 state 는 White


class VisualizationMixin:
    """RViz 시각화 메서드 묶음. StateMachine 에 다중 상속으로 mix-in 한다."""

    @staticmethod
    def _speed_to_color(vx, vx_min, vx_max):
        """속도 값을 (r, g, b) 그라데이션으로 변환 — 느리면 빨강, 빠르면 초록.

        vx_min == vx_max 인 경우 (모든 속도 동일) 중간값(0.5) 사용.
        """
        if vx_max > vx_min:
            t = (vx - vx_min) / (vx_max - vx_min)
        else:
            t = 0.5
        r = max(0.0, min(1.0, 1.0 - 2.0 * (t - 0.5)))
        g = max(0.0, min(1.0, 2.0 * t))
        return r, g, 0.0

    def _pub_local_wpnts(self, wpts):
        # 1) 이전 마커들 모두 삭제
        del_mrks = MarkerArray()
        del_mrk = Marker()
        del_mrk.header.stamp = self.get_clock().now().to_msg()
        del_mrk.action = Marker.DELETEALL
        del_mrks.markers.append(del_mrk)
        self.vis_loc_wpnt_pub.publish(del_mrks)

        # 2) local waypoints 메시지 + 위치 sphere 마커 생성
        loc_wpnts = WpntArray()
        loc_wpnts.wpnts = wpts
        loc_wpnts.header.stamp = self.get_clock().now().to_msg()
        loc_wpnts.header.frame_id = "map"

        vx_vals = [wpnt.vx_mps for wpnt in loc_wpnts.wpnts]
        vx_min = min(vx_vals) if vx_vals else 0.0
        vx_max = max(vx_vals) if vx_vals else 1.0

        loc_markers = MarkerArray()
        for i, wpnt in enumerate(loc_wpnts.wpnts):
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.type = mrk.SPHERE
            mrk.scale.x = 0.15
            mrk.scale.y = 0.15
            mrk.scale.z = 0.15
            mrk.color.a = 1.0
            mrk.color.r, mrk.color.g, mrk.color.b = self._speed_to_color(
                wpnt.vx_mps, vx_min, vx_max
            )
            mrk.id = i
            mrk.pose.position.x = wpnt.x_m
            mrk.pose.position.y = wpnt.y_m
            mrk.pose.position.z = wpnt.z_m
            mrk.pose.orientation.w = 1
            loc_markers.markers.append(mrk)

        self.loc_wpnt_pub.publish(loc_wpnts)
        self.vis_loc_wpnt_pub.publish(loc_markers)

        # 3) 속도 cylinder 마커 (높이 = 속도, 색상은 sphere와 동일)
        VEL_SCALE = 0.1317
        vel_markers = MarkerArray()
        for i, wpnt in enumerate(loc_wpnts.wpnts):
            mrk = Marker()
            mrk.header.frame_id = "map"
            mrk.header.stamp = self.get_clock().now().to_msg()
            mrk.type = Marker.CYLINDER
            mrk.id = i
            mrk.scale.x = 0.1
            mrk.scale.y = 0.1
            height = max(wpnt.vx_mps * VEL_SCALE, 0.02)
            mrk.scale.z = height
            mrk.color.a = 0.7
            mrk.color.r, mrk.color.g, mrk.color.b = self._speed_to_color(
                wpnt.vx_mps, vx_min, vx_max
            )
            mrk.pose.position.x = wpnt.x_m
            mrk.pose.position.y = wpnt.y_m
            mrk.pose.position.z = wpnt.z_m + height * 0.5
            mrk.pose.orientation.w = 1
            vel_markers.markers.append(mrk)
        self.vis_loc_vel_pub.publish(vel_markers)

    def _publish_target_marker(self, publisher, targets, *, color_b=0.0, color_g=0.0):
        """target 좌표(첫 원소)를 마커로 발행. targets가 비어있으면 DELETEALL."""
        marker = Marker()
        if len(targets) != 0:
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.b = color_b
            marker.color.g = color_g
            marker.pose.position.x = targets[0].x_m
            marker.pose.position.y = targets[0].y_m
            marker.pose.orientation.w = 1
        else:
            marker.action = Marker.DELETEALL
        publisher.publish(marker)

    def visualize_state(self, state: str):
        """현재 state 를 트랙 옆에 색상 sphere + 텍스트로 시각화."""
        # 첫 호출 시 트랙 왼쪽 옆 위치 한 번 계산 (이후 동일 위치 재사용)
        if self.first_visualization:
            self._compute_visualization_anchor()

        # State sphere 마커 (state 별 고정 색상)
        r, g, b = STATE_COLORS.get(state, _STATE_COLOR_DEFAULT)
        mrk = Marker()
        mrk.type = mrk.SPHERE
        mrk.id = 1
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.color.a = 1.0
        mrk.color.r = r
        mrk.color.g = g
        mrk.color.b = b
        mrk.pose.position.x = self.x_viz
        mrk.pose.position.y = self.y_viz
        mrk.pose.position.z = 0
        mrk.pose.orientation.w = 1
        mrk.scale.x = 1
        mrk.scale.y = 1
        mrk.scale.z = 1
        self.state_mrk.publish(mrk)

        # waypoint source + battery voltage 텍스트 마커
        self._publish_state_text_marker()

    def _compute_visualization_anchor(self):
        """첫 시각화 시 트랙 옆(자차 좌측 normal 방향) 시각화 anchor 좌표 계산."""
        self.first_visualization = False
        x0 = self.cur_gb_wpnts.list[0].x_m
        y0 = self.cur_gb_wpnts.list[0].y_m
        x1 = self.cur_gb_wpnts.list[1].x_m
        y1 = self.cur_gb_wpnts.list[1].y_m
        # normal 벡터 = 트랙 진행방향에 수직, 길이는 좌측 트랙폭의 1.25배
        xy_norm = (
            -np.array([y1 - y0, x0 - x1])
            / np.linalg.norm([y1 - y0, x0 - x1])
            * 1.25 * self.cur_gb_wpnts.list[0].d_left
        )
        self.x_viz = x0 + xy_norm[0]
        self.y_viz = y0 + xy_norm[1]

    def _publish_state_text_marker(self):
        """state sphere 위쪽에 wpnts_src + battery voltage 텍스트 마커 발행."""
        # 두 줄 텍스트를 동일 폭으로 가운데 정렬
        wpnt_src_str = str(self.local_wpnts_src).replace("StateType.", "")
        voltage_str = f"{self.cur_volt:.1f}V" if hasattr(self, 'cur_volt') else "~V"
        max_len = max(len(wpnt_src_str), len(voltage_str)) + 4
        wpnt_centered = wpnt_src_str.center(max_len)
        voltage_centered = voltage_str.center(max_len)

        text_mrk = Marker()
        text_mrk.type = Marker.TEXT_VIEW_FACING
        text_mrk.id = 2  # sphere 마커(id=1) 와 분리
        text_mrk.header.frame_id = "map"
        text_mrk.header.stamp = self.get_clock().now().to_msg()
        text_mrk.pose.position.x = self.x_viz
        text_mrk.pose.position.y = self.y_viz
        text_mrk.pose.position.z = 1.5  # sphere 위쪽
        text_mrk.pose.orientation.w = 1
        text_mrk.scale.z = 0.2
        text_mrk.color.r = 0.0  # 검정
        text_mrk.color.g = 0.0
        text_mrk.color.b = 0.0
        text_mrk.color.a = 1.0
        text_mrk.text = f"{wpnt_centered}\n {voltage_centered}"
        self.state_wpnts_src_marker.publish(text_mrk)

    def publish_not_ready_marker(self):
        """저전압 등 차량이 준비되지 않았음을 알리는 큰 빨간 텍스트 마커."""
        mrk = Marker()
        mrk.type = mrk.TEXT_VIEW_FACING
        mrk.id = 1
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.color.a = 1.0
        mrk.color.r = 1.0
        mrk.color.g = 0.0
        mrk.color.b = 0.0
        # 트랙 중앙에 publish 하여 잘 보이도록
        mrk.pose.position.x = np.mean([wpnt.x_m for wpnt in self.cur_gb_wpnts.list])
        mrk.pose.position.y = np.mean([wpnt.y_m for wpnt in self.cur_gb_wpnts.list])
        mrk.pose.position.z = 1.0
        mrk.pose.orientation.w = 1
        mrk.scale.x = 4.69
        mrk.scale.y = 4.69
        mrk.scale.z = 4.69
        mrk.text = "BATTERY TOO LOW!!!"
        self.emergency_pub.publish(mrk)
