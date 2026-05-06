# MIGRATION — ROS1 Noetic → ROS2 Jazzy 단계적 마이그레이션 (SH)

이 문서는 `ICRA2026_HJ` (ROS1 Noetic) 코드베이스를 ROS2 Jazzy 로 단계적
포팅하기 위한 분석 + 계획 + 진행 트래킹 문서다.

원본 ROS1 워크스페이스는 그대로 유지하고, 별도 새 워크스페이스
(`~/unicorn_ws/ICRA2026_SH_ros2/`) 에 패키지 단위로 포팅한다.

---

## 결정 사항

| 항목 | 값 |
|---|---|
| **타겟 distro** | ROS2 Jazzy (LTS, Ubuntu 24.04 + Python 3.12) |
| **전략** | 단계적 포팅 (Big bang ❌, ros1_bridge ❌) |
| **워크스페이스** | `~/unicorn_ws/ICRA2026_SH_ros2/` (호스트 직접, Docker 미사용) |
| **외부 의존** | cartographer / GLIM / livox / vesc / blink 등 — **skip** (ROS2 포팅 별도 작업) |
| **검증 환경** | 호스트의 `/opt/ros/jazzy` (이미 설치) — 단계마다 `colcon build` + `ros2` 명령으로 확인 |

---

## 코드베이스 footprint (원본 ROS1)

| 항목 | 수치 |
|---|---|
| ROS1 패키지 (package.xml) | 57 |
| Python rospy import 파일 | 162 |
| Python rospy 호출 라인 | 5,525 |
| C++ roscpp 사용 파일 | 33 |
| Custom msg/srv/action | 32 (5 패키지) |
| .launch 파일 | 91 (XML) |
| package.xml format=2 / format=3 | 45 / 4 |

### rospy 자주 쓰는 API (변환 매핑)

| ROS1 (rospy) | 사용 횟수 | ROS2 (rclpy) |
|---|---|---|
| `rospy.get_param` | 1,361 | `node.declare_parameter` + `node.get_parameter` |
| `rospy.loginfo` | 971 | `node.get_logger().info(...)` |
| `rospy.Subscriber` | 548 | `node.create_subscription(msg, topic, cb, qos)` |
| `rospy.Publisher` | 447 | `node.create_publisher(msg, topic, qos)` |
| `rospy.Time` | 411 | `rclpy.time.Time` / `node.get_clock()` |
| `rospy.wait_for_message` | 211 | rclpy 미지원 — 직접 헬퍼 작성 또는 `wait_for_message` 라이브러리 |
| `rospy.init_node` | 129 | `rclpy.init()` + `Node('name')` |
| `rospy.spin` | 56 | `rclpy.spin(node)` |

### 외부 의존 — ROS2 호환성

| 의존 | ROS2 처리 |
|---|---|
| roscpp / rospy | → rclcpp / rclpy (이름 + API 변경) |
| std_msgs, nav_msgs, geometry_msgs, sensor_msgs, ackermann_msgs, visualization_msgs | ✅ ROS2도 그대로 |
| tf2_ros, tf2_geometry_msgs | ✅ ROS2도 그대로 |
| **dynamic_reconfigure** (18 패키지 사용) | ⚠️ ROS2엔 없음 — `parameter_event` subscriber 패턴으로 재구현 |
| message_generation, message_runtime | → `rosidl_default_generators` / `rosidl_default_runtime` |
| pcl_ros, cv_bridge, image_transport | ✅ ROS2 버전 있음 |
| **cartographer_ros, GLIM, livox, vesc** | ⚠️ **skip** (외부 의존 — 별도 작업) |

---

## 마이그레이션 순서 (의존 그래프)

```
Phase A — 기반 (ROS 의존 적은 것부터)
  1. f110_msgs           — 메시지 정의만, 의존 거의 없음 (가장 쉬움)
  2. f110_utils libs     — frenet_conversion, vel_planner_25d 등 (메시지 의존)

Phase B — 단순 노드 (검증 환경 구축)
  3. obstacle_publisher  — 단순 publisher 노드, 검증 쉬움
  4. fake_odom_publisher — stack_master/scripts/, 검증 시뮬에 필수
  5. global_republisher  — gb_optimizer/global_trajectory_publisher.py

Phase C — 핵심 노드 (우리가 정리한 것)
  6. **state_machine** (가장 가치 있는 작업)
     - path_checker.py: ROS 의존 없는 순수 함수 → 그대로 사용 가능
     - waypoint_data.py: rospy.Subscriber 만 변경
     - state_machine_init.py + state_machine_callbacks.py: rospy → rclpy
     - state_machine_visualization.py: 메시지 import 만 변경
     - state_transitions.py + states.py: ROS 의존 없음 (거의)

Phase D — Planner / Controller / Perception
  7. spliner / recovery_spliner / sqp_planner / 3d_static_avoidance_node
  8. controller (controller_manager.py, Controller.py)
  9. prediction (gp_traj_predictor)

Phase E — 외부 의존 (별도 일정)
  10. cartographer_ros2 / GLIM ROS2 / livox_ros_driver2 ROS2 / vesc ROS2
      → 외부 패키지 ROS2 버전 사용 또는 별도 포팅
```

---

## Phase 별 작업 분량 추정

| Phase | 패키지 | 시간 |
|---|---|---|
| A | f110_msgs + f110_utils 핵심 | 1~2일 |
| B | obstacle_publisher + fake_odom + global_republisher | 1일 |
| C | **state_machine** (우리 핵심) | **2~3일** |
| D | spliner / controller / perception | 1~2주+ |
| E | 외부 의존 — skip (이번 작업 외) | — |
| **검증 가능 minimal (A~C)** | **약 1주** | |
| 풀 마이그레이션 (E 제외) | 2~4주 | |

---

## 워크스페이스 구조

```
~/unicorn_ws/
  ICRA2026_HJ/           ← ROS1 원본 (계속 유지, 변경 없음)
  ICRA2026_SH_ros2/      ← ROS2 새 ws (포팅 대상, SH 단독)
    src/
      f110_msgs/         ← Phase A-1
      frenet_conversion/ ← Phase A-2
      ...
      state_machine/     ← Phase C
    build/
    install/
    log/
```

원본은 절대 건드리지 않음. 포팅된 코드는 새 ws의 src/ 에 복사 + 수정.

---

## 위험 / 주의

| 항목 | 영향 | 대응 |
|---|---|---|
| `dynamic_reconfigure` ROS2 미지원 | sector_tuner_3d, dyn_statemachine 등 영향 | parameter_event subscriber 패턴 재구현 |
| `rospy.wait_for_message` 미지원 | state_machine 의 startup blocking | 직접 helper 작성 (timeout + topic 한 번 받기) |
| 외부 의존 skip | cartographer, GLIM, livox 등 사용 노드 빌드 안 됨 | 검증 외이므로 OK |
| `time` API 차이 | `rospy.Time.now()` → `node.get_clock().now()` | 호출자 변경 필요 |
| 메시지 타입 호환성 | f110_msgs 의 Wpnt 등 타입은 그대로 | CMakeLists 변경만 |

---

## 진행 트래킹

| Phase | 상태 | 비고 |
|---|---|---|
| MIGRATION.md 작성 | ✅ 완료 | |
| ROS2 워크스페이스 셋업 | ✅ 완료 | `~/unicorn_ws/ICRA2026_SH_ros2/` |
| **Phase A-1: f110_msgs 포팅** | ✅ **완료** | 18 msgs, OTWpntArray 의 `time` → `builtin_interfaces/Time` 만 변경. colcon build 7s |
| Phase A-2: f110_utils 라이브러리 (frenet_conversion 등) | pending | |
| Phase B: 단순 노드 (obstacle_publisher / fake_odom_publisher 등) | ⏳ 진행 중 | 첫 노드 포팅으로 패턴 확립 |
| Phase C: state_machine | pending | 우리 핵심 작업 |
| Phase D: planner / controller | pending | |

---

## Phase A-1 — f110_msgs 포팅 결과 (2026-05-04)

### 변경 내용
- `package.xml` format=2 (catkin) → format=3 (ament_cmake + rosidl_default_generators)
- `CMakeLists.txt` catkin → ament + `rosidl_generate_interfaces`
- `OTWpntArray.msg` 의 `time last_switch_time` → `builtin_interfaces/Time last_switch_time`
  (그 외 17개 메시지는 정의 그대로 사용)

### 검증
- `colcon build --packages-select f110_msgs` ✅ 7.13s
- `ros2 interface list | grep f110_msgs` → 18 메시지 모두 등록
- Python instantiate (`Wpnt`, `WpntArray`, `OTWpntArray`, `BehaviorStrategy`) ✅

### 발견 사항
- 호스트의 다른 ws (`~/creating_autonomous_car_ws/`) 에 같은 이름 `f110_msgs` 존재 → colcon override 경고. 우리 새 ws의 install 이 우선 (overlay). 동작 영향 없음.
