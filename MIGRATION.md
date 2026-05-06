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
| **Phase A-2a: frenet_conversion_msgs (srv 4개)** | ✅ **완료** | 별도 인터페이스 패키지로 분리. 4 srv 등록 + Python instantiate 검증 |
| **Phase A-2b: frenet_conversion (Python lib + 서버)** | ✅ **완료** | Python lib (404L) ROS-free + Python service server. C++ 미포팅 (분기 부채 청산). 16/16 단위 테스트 + 4 service 런타임 검증 |
| **Phase B-1: fake_odom_publisher 포팅** | ✅ **완료** | 108→125 라인 + 100 라인 순수 함수 분리. 13/13 단위 테스트 + ros2 런타임 20Hz 발행 검증 |
| **Phase B-2: random_obstacle_publisher 포팅** | ✅ **완료** | 109→160 라인 + 110 라인 순수 함수. 12/12 단위 테스트 + ros2 런타임 검증 (5 sectors, ids 정확) |
| **Phase B-5: frenet_odom_republisher 포팅** | ✅ **완료** | 240L C++ → Python (160L 노드 + 20L 헬퍼). frenet_conversion lib 의 get_frenet_odometry 메서드 신규 추가. 6/6 + 3/3 단위 테스트 + ros2 런타임 검증 |
| **Phase B-6: global_republisher 포팅** | ✅ **완료** | 173L Python → 220L (lib 분리 80L + 노드 240L). ROS1 → ROS2 dict↔msg 변환 + legacy 필드 cleanup (seq/secs/nsecs). 7/7 단위 테스트 + ROS1 실제 JSON (gazebo_wall_2) 로드 + 12 토픽 0.5Hz 발행 검증 |
| **Phase B-3: obstacle_publisher 포팅** | ✅ **완료** | 252L Python (frenet service client) → 250L 노드 + 40L 순수. ROS2 service client 패턴 (wait_for_message + spin_until_future_complete + call_async + done_callback) 확립. 12/12 단위 테스트 + 3-노드 통합 검증 (global_republisher + frenet_server + obstacle_publisher) → /tracking/obstacles 50Hz 정확 |
| Phase B-3+: obstacle_publisher (frenet 의존) / global_republisher | pending | obstacle_publisher 본체는 frenet_conversion 서비스에 강하게 묶여 있어 A-2 후 진행 |
| **Phase C-1: state_machine 분석** | ✅ **완료** | 3500L 포팅 대상 식별 + sub-phase 분할 (C-2..C-6) |
| **Phase C-2: 순수 모듈 + 67 pytest** | ✅ **완료** | path_checker / state_transitions / states / states_types / waypoint_data 옮김 (rospy → logging). 67/67 pass |
| **Phase C-3: mixin 모듈 (init / callbacks / viz / smart_helper)** | ✅ **완료** | rospy → rclpy 변환. 4 mixin smoke import + 기존 67 pytest 회귀 검증 |
| **Phase C-4: 메인 노드 (3d_state_machine_node, 1212L)** | ✅ **완료** | rospy → rclpy + Node 다중 상속 + main(). MRO 검증 + 회귀 67 pytest + smoke (의도된 launch param fail 까지) |
| **Phase C-5: dynamic_*_server → ROS2 native parameter 통합** | ✅ **완료** | 메인 노드의 `add_on_set_parameters_callback` 으로 흡수. dyn_statemachine/* 12 파라미터 ros2 param set 호환 |
| Phase C-6: 4-노드 통합 launch + 시나리오 검증 | pending | |
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

---

## Phase B-1 — fake_odom_publisher 포팅 결과 (2026-05-04)

원본: `ICRA2026_HJ/stack_master/scripts/fake_odom_publisher.py` (108 라인, HJ 작성).
포팅: `ICRA2026_SH_ros2/src/fake_odom_publisher/` (ament_python 패키지).

### 변환 매핑 (검증된 패턴)

| ROS1 (rospy) | ROS2 (rclpy) |
|---|---|
| `rospy.init_node("name", anonymous=True)` | `class Node(...): super().__init__("name")` |
| `rospy.get_param("~map", default)` | `declare_parameter("map", default)` + `get_parameter("map").get_parameter_value().string_value` |
| `rospy.Publisher(topic, MsgType, queue_size=10)` | `self.create_publisher(MsgType, topic, 10)` |
| `rospy.Rate(50)` + `while not rospy.is_shutdown(): ... rate.sleep()` | `self.create_timer(1.0/50, self._tick)` + `rclpy.spin(node)` |
| `rospy.Time.now()` | `self.get_clock().now().to_msg()` |
| `tf.transformations.quaternion_from_euler(0,0,psi)` | `tf_transformations` 외부 의존 회피 — 직접 sin/cos 으로 yaw quaternion 계산 (`raceline.yaw_to_quaternion`) |
| `rospy.loginfo(...)` | `self.get_logger().info(...)` |
| `if __name__ == "__main__": Node()` | `def main(args=None): rclpy.init(args=args); rclpy.spin(node); rclpy.shutdown()` |
| `package.xml` format=2 + `CMakeLists.txt` (catkin) | `package.xml` format=3 + `setup.py` + `setup.cfg` + `resource/<pkg>` (ament_python) |

### 구조 개선 (포팅 + 동시 진행)

원본 ROS1 노드는 모든 수치 로직이 `__init__` + `run()` 안에 묶여 있어 단위 테스트 불가능.
포팅하면서 **순수 함수만 분리**:

- `raceline.py` — `Waypoint` / `Pose3D` dataclass + `find_segment_index`,
  `interpolate_pose`, `yaw_to_quaternion`, `waypoints_from_dicts` (ROS 의존 0)
- `fake_odom_publisher.py` — Node 클래스 (파라미터 / 발행자 / 타이머만)
- `test/test_raceline.py` — pytest 13개 (보간, slope vz, quaternion norm 검증)

### 검증

- `colcon build --packages-select fake_odom_publisher` ✅ 0.8s
- `pytest test/test_raceline.py` ✅ 13/13 pass
- `ros2 run fake_odom_publisher fake_odom_publisher --ros-args -p rate:=20.0` 기동
  → `ros2 topic hz /glim_ros/base_odom` 결과 **20.001 Hz, std 0.1ms** (window=30)
  → `ros2 topic echo --once` 으로 position xyz / orientation quaternion / twist vx,vz 정상

### 부수 발견

- 호스트 환경의 `~/.cyclonedds.xml` 이 192.168.70.x (차량 네트워크) hard-bind →
  현재 머신 (192.168.50.x) 에서 노드 기동 시 `does not match an available interface` 에러.
  **해결**: `unset CYCLONEDDS_URI` (검증 셸 한정) → cyclone 자동 인터페이스 선택. 사용자 시스템 설정 미수정.
- 원본의 `map` 기본값 `gazebo_wall_3d_rc_car_10th_timeoptimal` 은 디렉터리명이 아니라 csv 파일명 → 디렉터리는 `gazebo_wall_2`. 포팅 노드 기본값을 `gazebo_wall_2` 로 수정 (호환성: `--ros-args -p map:=...` 로 override 가능).

---

## Phase B-2 — random_obstacle_publisher 포팅 결과 (2026-05-04)

원본: `ICRA2026_HJ/f110_utils/nodes/random_obstacle_publisher` (109L).
포팅: `ICRA2026_SH_ros2/src/random_obstacle_publisher/`.

### 핵심 변환 (B-1 패턴 + 신규)

| 항목 | 원본 ROS1 | 포팅 ROS2 |
|---|---|---|
| 무한 루프 | `__init__` 안 `while not is_shutdown(): rate.sleep()` | `create_timer(1/rate, _tick)` + `rclpy.spin(node)` |
| ready 대기 | startup `while (not has_traj or not has_odom): sleep` | `_tick` 안 가드 + `if not _initialized: build_once()` |
| 단일 발행 | rate.sleep 안에서 publish | timer callback 안에서 publish |

### 구조 개선

- **dead imports 정리**: `LapData`, `OccupancyGrid`, `Bool` — 원본에서 import 만 하고 사용 0 → 미포팅
- **순수 함수 분리** (`obstacle_geometry.py`, 110L): `WaypointSpec` / `ObstacleSpec` dataclass + `get_closest_point_on_traj`, `generate_random_obstacle`, `build_sector_obstacles`, `select_obstacles_in_lookahead`
- 노드 (`random_obstacle_publisher.py`, 160L): 파라미터 + 콜백 + tick 만
- 12개 pytest (sector ids, d-width, slope max_d clamp, lookahead wrap, deterministic seed)

### 검증

- `colcon build` ✅ 0.8s
- `pytest test/test_obstacle_geometry.py` ✅ 12/12
- `ros2 run` + fake `/global_waypoints` + `/car_state/odom_frenet` (ros2 topic pub) 으로
  → `Generated 5 random obstacles (final_s=10.00m)` ✓
  → `/obstacles` 메시지 5개, **ids = 0,1,2,3,4** (sector index 정확)
  → `s_end - s_start = 0.3` (obstacle_length), `d_left - d_right = 0.2` (obstacle_width) ✓
  → `topic hz` = 20 Hz (노드 1개 기준)

### 주요 발견 — **f110_msgs 메시지 hash 충돌** (모든 ROS2 검증에 영향)

`/global_waypoints` topic 의 publisher (`ros2 topic pub`) 와 subscriber (random_obstacle_publisher 노드) 가
**서로 다른 hash 의 f110_msgs/WpntArray** 를 사용 → DDS 가 메시지를 **silently drop**:

```
Publisher  hash: RIHS01_745ac277...  (다른 ws의 f110_msgs)
Subscriber hash: RIHS01_c3c62139...  (우리 ws의 f110_msgs)
```

원인: 호스트의 다른 ws (`~/creating_autonomous_car_ws/`) 가 같은 이름의 `f110_msgs` 를 가지고 있고,
사용자의 `~/.bashrc` 또는 환경 어딘가에서 그 ws install 이 active. 우리 ws 의 `install/setup.bash` 를
source 하지 않은 환경에서 `ros2 topic pub` 를 띄우면 **다른 ws 의 메시지 정의를 사용** → hash mismatch.

**해결**: 모든 ros2 명령 (pub, run) 에 우리 ws install 을 source:
```bash
source /opt/ros/jazzy/setup.bash
source ~/unicorn_ws/ICRA2026_SH_ros2/install/setup.bash
unset CYCLONEDDS_URI
ros2 topic pub ...
```

이 mismatch 가 잡히지 않은 이유: nav_msgs / std_msgs 같은 ROS2 native 는 어디서 source 해도 동일 hash —
B-1 (fake_odom_publisher) 검증은 `Odometry` 만 발행했기 때문에 발견되지 않았음. 첫 f110_msgs 사용 노드인 B-2 에서 발견.

→ **이후 f110_msgs 사용 노드 검증 시 항상 우리 ws install/setup.bash source 필수**.

### 부수 가드

원본은 빈/invalid wpnts 메시지에 대한 가드 없음. 포팅 시 추가:
```python
if not msg.wpnts:
    return
if wpnts[-1].s_m <= 0.0:
    return
```
이유: ros2 환경의 first-message race 방어 + IndexError 방지.

---

## Phase A-2 — frenet_conversion 포팅 결과 (2026-05-04)

원본은 **C++ 와 Python 두 분기 구현이 공존** (README 의 "Python TODO" 가 누군가에 의해 자체 구현으로 전환된 것). 두 구현이 다음과 같이 분리:
- **C++ lib + 서버** (`libs/frenet_conversion/src/frenet_conversion.cc` + `nodes/frenet_conversion_server/`): 다른 노드가 ROS service 로 호출
- **Python lib** (`libs/frenet_conversion/src/frenet_converter/frenet_converter.py`): import 사용. 3D 메서드 + height filter + boundary raycast + e_psi 등 추가 기능 풍부.

이번 포팅은 **Python 만 유지** (옵션 C). 분기 부채 청산 + ROS2 lib 가 단일 구현으로 통합.

### A-2a — `frenet_conversion_msgs` (srv 4개)

별도 인터페이스 패키지로 분리 (사용자가 lib 의존 없이 srv 만 가져갈 수 있게).

| srv | Request | Response |
|---|---|---|
| Glob2Frenet | x, y, z | s, d, idx |
| Glob2FrenetArr | x[], y[], z[] | s[], d[], idx[] |
| Frenet2Glob | s, d | x, y, z |
| Frenet2GlobArr | s[], d[] | x[], y[], z[] |

빌드: `colcon build` ✅ 4.5s. `ros2 interface list` → 4 srv 등록.

### A-2b — `frenet_conversion` (Python lib + Python service server)

**`frenet_converter.py` (~340L 순수)**:
- 원본 Python lib 그대로 + ROS 의존 (`_load_track_bounds` 의 `rospy.wait_for_message`) **제거**.
- 외부 호출자 (서버 노드) 가 `set_track_bounds_from_markers(markers)` 직접 호출하도록 책임 이전.
- `get_approx_s_3d_with_idx` 메서드 신규 추가 — service idx 응답 위해 인덱스도 함께 반환.
- 그 외 알고리즘 (build_raceline, get_frenet_3d, height filter, boundary raycast, rotational search, perpendicular projection) 모두 원본 그대로.

**`frenet_converter_server.py` (~180L)**:
- `/global_waypoints` 구독 → FrenetConverter 빌드 (try/except 가드 — invalid 메시지에 노드가 죽지 않게).
- `/trackbounds/markers` 구독 → set_track_bounds_from_markers (원본 C++ 서버에는 없는, Python lib 의 기능).
- 4 service 제공 (Glob2Frenet[Arr], Frenet2Glob[Arr]). PerceptionOnly=true 면 `_perception` 접미사.

### 검증

- `colcon build --packages-select frenet_conversion` ✅ 0.8s
- `pytest test/test_frenet_converter.py` ✅ 16/16 (build_raceline 직선/3D 슬로프, get_cartesian, frenet round-trip, get_approx_s 2D/3D, set_track_bounds, e_psi 등)
- `ros2 service list` → 4 service 등록 (`/convert_frenet2glob_service`, `/convert_glob2frenet_service`, `/convert_frenet2globarr_service`, `/convert_glob2frenetarr_service`)
- 직선 트랙 (10 wpnts, x=0..9) 발행 후 service call 결과 (모두 정확):
  - `Frenet2Glob(s=5.0, d=0.5)` → `(x=5.0, y=0.5, z=0.0)` ✓
  - `Glob2Frenet(x=3.0, y=0.5, z=0)` → `(s=3.0, d=0.5, idx=3)` ✓
  - `Frenet2GlobArr([(1,0),(5,0.5),(8,-0.3)])` → `[(1,0),(5,0.5),(8,-0.3)]` ✓
  - `Glob2FrenetArr([(2,0,0),(5,0.3,0)])` → `s=[2,5], d=[0,0.3], idx=[2,5]` ✓
- PerceptionOnly=true → service 이름에 `_perception` 접미사 적용 ✓

### 신규 ROS2 패턴 (이후 재사용)

| 카테고리 | ROS1 | ROS2 |
|---|---|---|
| Service 제공 | `nh.advertiseService(name, &Class::Cb, this)` | `self.create_service(SrvType, name, self._cb)` |
| Service callback 시그니처 | `bool Cb(SrvType::Request &req, SrvType::Response &res)` | `def _cb(self, request, response):` (response 갱신 후 return response) |
| Service 클라이언트 호출 | `rospy.ServiceProxy(name, SrvType)(req)` | `self.create_client(SrvType, name)` + `call_async(req)` (별도 Phase 시 사용) |
| 인터페이스 패키지 분리 | `add_service_files` + `generate_messages` 같은 패키지 | 권장: `_msgs` / `_interfaces` 별도 패키지 (rosidl_default_generators) |

### 부수 가드 (포팅 시 추가)

`_on_global_traj` 에 try/except — `CubicSpline` 이 strictly-increasing 위반으로 throw 시 노드가 죽지 않게:

```python
try:
    converter = FrenetConverter(x, y, z)
except (ValueError, IndexError) as e:
    self.get_logger().warn(f"... ignoring this msg")
    return
```

이유: ros2 topic pub default-empty 메시지가 wpnts 5개 이상 가지고 있어도 모든 s_m=0 일 수 있고, scipy CubicSpline 이 `x must be strictly increasing` 으로 ValueError 발생.

### 실행 환경 잔재 (운영 시 주의)

호스트의 `~/catkin_ws/devel/.../frenet_conversion_server_node` (ROS1 노드) 가 4개 살아있는 것을 발견. 사용자의 다른 ROS1 작업 잔재로 보이며 우리 ROS2 검증과는 별개 (DDS / ROS_DOMAIN 격리). 죽이지 않고 그대로 두었음.

---

## Phase B-5 — frenet_odom_republisher 포팅 결과 (2026-05-04)

원본: `ICRA2026_HJ/f110_utils/nodes/frenet_odom_republisher` (C++, 240L).
포팅: `ICRA2026_SH_ros2/src/frenet_odom_republisher/` (Python).

C++ 원본은 `frenet_conversion::FrenetConverter` C++ lib 를 link 하는데, 우리는
Phase A-2 에서 C++ lib 미포팅 결정 → **Python 으로 재작성** + 우리 frenet_conversion
Python lib 를 import 사용.

### 신규 추가 (frenet_conversion lib 에)

원본 C++ 의 `FrenetConverter::GetFrenetOdometry(x, y, z, yaw, vx_body, vy_body, ...)`
한 함수가 **변환 + idx + frenet velocity** 까지 한 번에 계산. Python lib 에는
이 메서드가 없었음 — A-2b 에 메서드 신규 추가:

```python
def get_frenet_odometry(self, x, y, z, yaw, vx_body, vy_body):
    s_arr, idx_arr = self.get_approx_s_3d_with_idx(...)
    s, d = self.get_frenet_coord(...)
    psi_track = self.waypoints_psi[idx_arr[0]]
    delta_psi = yaw - psi_track
    vs = vx_body * cos(delta_psi) - vy_body * sin(delta_psi)
    vd = vx_body * sin(delta_psi) + vy_body * cos(delta_psi)
    return s, d, vs, vd, idx
```

원본 C++ 의 `CalcFrenetVelocity` 공식을 그대로 옮김 (`R(yaw - psi_track) * (vx_body, vy_body)`).
주의: 원본은 wpnt.psi_rad 필드 (사용자가 별도 제공) 를 사용했으나, 우리 Python lib 는 받지 않음
→ spline derivative 에서 자동 계산한 `waypoints_psi` 사용 (실용상 동일).

### 노드 구조

| 부분 | 라인 | 책임 |
|---|---|---|
| `transforms.py` | 20 | `quaternion_to_yaw` 순수 함수 (원본의 tf::Quaternion + getRPY 대체) |
| `frenet_odom_republisher.py` | 160 | Node — sub 4개 + pub 2개 + odom 변환 |

토픽 매핑 (원본 launch remap 그대로):
| sub | / pub | 토픽 |
|---|---|---|
| sub | `/odom` (Odometry, launch remap → `/car_state/odom`) |
| sub | `/global_waypoints` (WpntArray) → GB FrenetConverter 빌드 |
| sub | `/planner/avoidance/smart_static_otwpnts` (OTWpntArray) → Fixed FrenetConverter 빌드 |
| sub | `/trackbounds/markers` (MarkerArray, 한 번만) → 두 converter 에 set_track_bounds |
| pub | `/odom_frenet` → `/car_state/odom_frenet` (GB frame) |
| pub | `/odom_frenet_fixed` → `/car_state/odom_frenet_fixed` (Smart Static frame) |

### 미포팅 (의도)

- **Interactive marker "Force Full Search" 버튼** — ROS2 interactive_markers 가 별도 패키지 의존 + 부수 디버깅 기능. 보류 (필요 시 향후 ros2 service 또는 cli 로 재구현).
- **C++ FrenetConverter::ForceFullSearch** — 위와 짝. 우리 Python lib 의 `get_approx_s_3d` 는 매 호출마다 search 하므로 first-call full search 개념 자체가 없음 (성능 차이 있을 수 있으나 실용상 OK).

### 검증

- `colcon build --packages-select frenet_odom_republisher` ✅ 0.8s
- `pytest test/test_transforms.py` ✅ 6/6 (quaternion_to_yaw)
- `pytest test/test_frenet_converter.py` ✅ **19/19** (16 + 신규 3 — `get_frenet_odometry` aligned / perpendicular yaw / lateral offset)
- ros2 run + fake `/global_waypoints` (10 wpnts, x=0..9) + fake `/odom` (pos=(3,0.5,0), yaw=0, vx=2.0):
  - `/odom_frenet` 출력: position.x=s=**3.0**, position.y=d=**0.5**, twist.linear.x=vs=**2.0**, twist.linear.y=vd=**0.0**, child_frame_id=**'3'** ✓
  - topic hz: 5.0 Hz (odom 입력 그대로), std 0.25ms

### 신규 ROS2 패턴

| 카테고리 | ROS1 (C++) | ROS2 (Python) |
|---|---|---|
| `tf::Quaternion` + `getRPY` | tf 패키지 의존 | 직접 atan2 계산 (transforms.py) |
| Launch `<remap from="/odom" to="/car_state/odom"/>` | xml | `Node(remappings=[('/odom', '/car_state/odom')])` |
| 한 번 받고 unsub (`trackbounds_sub_.shutdown()`) | C++ Subscriber.shutdown | Python flag (`if self._has_track_bounds: return`) — destroy_subscription 도 가능하나 단순화 |

---

## Phase B-6 — global_republisher 포팅 결과 (2026-05-04)

원본 ROS1: `planner/gb_optimizer/src/global_trajectory_publisher.py` (173L) +
`readwrite_global_waypoints.py` (149L). 노드 이름이 `global_republisher` 인데
패키지명 / 파일명은 `gb_optimizer/global_trajectory_publisher` 라 처음에 못 찾음.
포팅: `ICRA2026_SH_ros2/src/global_republisher/`.

### 책임

매핑 phase 에서 만든 `<map>/global_waypoints.json` 을 한 번 로드한 뒤, 0.5 Hz 로
**12개 트랙 관련 토픽** sticky 발행 + **`track_length` 파라미터 set**:

| 토픽 | 메시지 |
|---|---|
| `/global_waypoints` | WpntArray |
| `/global_waypoints/markers` | MarkerArray |
| `/global_waypoints/shortest_path` + `/markers` | WpntArray + MarkerArray |
| `/centerline_waypoints` + `/markers` | WpntArray + MarkerArray |
| `/trackbounds/markers` | MarkerArray |
| `/map_infos`, `/estimated_lap_time` | String, Float32 |
| `/lattice_viz`, `/global_waypoints/vel_markers` | MarkerArray (옵셔널) |
| `/global_waypoints/vel_markers_tuned` | 매 tick 새로 빌드 (vx_mps 기반) |

자기가 발행하는 토픽도 sub → 외부 노드 (vel_planner 등) 가 같은 토픽 발행하면 그것을 대신 republish.

### 미포팅 (의도)

- **`write_global_waypoints`** — 매핑 phase 책임. 검증 / state_machine 진입에 불필요.
- **`stack_master/maps/<name>/global_waypoints.json` 자동 경로 탐색** — `stack_master` 미포팅이라 `rospkg.RosPack().get_path("stack_master")` 사용 불가. 대안: 파라미터 `map_path` (절대경로) 또는 `map` (이름) → fallback `~/unicorn_ws/ICRA2026_HJ/stack_master/maps/<map>/global_waypoints.json` (B-1 패턴).

### **ROS1 → ROS2 호환성 발견 — JSON legacy 필드 (다른 phase 도 영향 가능)**

원본은 `rospy_message_converter` 사용. ROS2 native `rosidl_runtime_py.set_message_fields`
로 대체했더니 **JSON 안의 ROS1 필드 두 가지가 ROS2 에서 fail**:

| ROS1 필드 | ROS2 |
|---|---|
| `std_msgs/Header.seq` | **제거됨** |
| `time.secs`, `time.nsecs` | **`builtin_interfaces/Time.sec`, `nanosec` 으로 이름 변경** |

해결: `_strip_legacy_fields` 재귀 함수로 dict 사전 정리:
```python
_LEGACY_FIELDS_TO_STRIP = {"seq"}
_LEGACY_FIELDS_TO_RENAME = {"secs": "sec", "nsecs": "nanosec"}
```

이게 **ROS1 → ROS2 마이그레이션의 일반적 함정**. 직렬화된 ROS1 데이터 (rosbag, JSON, YAML 등) 를 ROS2 에서 로드할 땐 같은 cleanup 필요.

### 신규 ROS2 패턴

| 카테고리 | ROS1 | ROS2 |
|---|---|---|
| dict ↔ msg 변환 | `rospy_message_converter.message_converter` | `rosidl_runtime_py.set_message_fields` (ROS2 native) |
| 패키지 경로 찾기 | `rospkg.RosPack().get_path("pkg")` | `ament_index_python.packages.get_package_share_directory("pkg")` |
| 동적 parameter set | `rospy.set_param("name", value)` | `declare_parameter` (이번엔 startup 만이라 declare 로 충분) |

### 검증

- `colcon build --packages-select global_republisher` ✅ 0.8s
- `pytest test/test_readwrite.py` ✅ **7/7** (file_not_found, minimum dict, centerline, track_length, optional vel_markers missing/present, **실제 ROS1 JSON gazebo_wall_2 로드**)
- ros2 run + 실제 ROS1 JSON (`gazebo_wall_2`) 로드 → `track_length=85.64m`, 12 토픽 발행
- `/global_waypoints` topic hz: **0.500 Hz**, std 0.18ms ✓
- `ros2 param get /global_republisher track_length` → `85.6356869406511` ✓
- 첫 wpnt 의 모든 3D 필드 (s_m, x_m, y_m, **z_m**, **mu_rad**, kappa_radpm, vx_mps) 정상

---

## Phase B-3 — obstacle_publisher 포팅 결과 (2026-05-04)

원본 ROS1: `f110_utils/nodes/obstacle_publisher/src/obstacle_publisher.py` (252L 본체).
포팅: `ICRA2026_SH_ros2/src/obstacle_publisher/`.

**첫 frenet service client 노드** — A-2 의 srv 정의가 실제 client 로 사용되는 첫 사례.

### 책임

선택한 트랙 (`min_curv` / `shortest_path` / `centerline` / `updated`) 위 raceline 을 따라
가짜 상대 차량 1대를 `vx_mps × speed_scaler` 속도로 주행시키며 `/tracking/obstacles`
(ObstacleArray) 50 Hz 발행. 시작 위치는 `start_s` 파라미터.

토픽:
- sub: `/global_waypoints` + 선택된 trajectory 토픽 + `/car_state/odom_frenet`
- pub: `/tracking/obstacles`, `/dummy_obstacle_markers`, `/opponent_waypoints` (25-tick 마다)
- service client: `convert_glob2frenetarr_service`, `convert_frenet2globarr_service`

### **신규 ROS2 패턴 — Service Client + startup 흐름** (이후 service 사용 노드들에 재사용)

ROS2 의 service client 사용은 **timer callback 안에서 동기 호출 (`spin_until_future_complete`) 호출 금지** ("Executor is already spinning" RuntimeError). 두 가지 패턴 조합:

**Startup phase (spin 시작 전, `__init__` 안)**:
```python
from rclpy.wait_for_message import wait_for_message

# 1. service 가 ready 될 때까지 대기
client.wait_for_service(timeout_sec=10.0)

# 2. 토픽 한 번 받기 (wait_for_message 가 내부적으로 spin)
ok, msg = wait_for_message(WpntArray, self, "/global_waypoints", time_to_wait=10.0)

# 3. 동기 service 호출 (call_async + spin_until_future_complete)
future = client.call_async(req)
rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
resp = future.result()

# 4. 그 다음에야 timer / 일반 콜백 등록
self.create_timer(...)
```

**Runtime phase (timer callback 안)**:
```python
def _tick(self):
    future = client.call_async(req)
    future.add_done_callback(self._on_done)  # 비동기로 처리

def _on_done(self, future):
    resp = future.result()
    # 응답 가공 + 발행
```

이 두 패턴이 ROS2 service client 사용의 표준. 원본 ROS1 코드는 `wait_for_message` + `ServiceProxy()` (동기) 라 별 신경 안 썼지만 ROS2 는 매우 strict.

### **신규 발견 — float64 필드에 Python int 넣기 (ROS1 → ROS2)**

`OpponentTrajectory.lap_count` 가 `float64` 인데 원본은 `lap_count = 2` (Python int). ROS1 은 implicit cast OK. ROS2 는 strict — `PyFloat_Check` assert fail로 **노드 abort**.

수정: `lap_count = 2.0` 명시.

이게 **ROS1 → ROS2 마이그레이션의 또 다른 함정** (B-6 의 legacy 필드 cleanup 와 별개). 직렬화된 데이터가 아니라도, 코드 안에서 필드 할당 시에도 strict 타입 체크.

### 미포팅 (의도)

- `dynamic_reconfigure` (`cfg/dyn_obs_publisher.cfg` + `dynamic_obs_pub_server.py`) — 원본 launch 에서 주석 처리되어 있어 dead code (별도 절차에서 옵션 a 선택).
- 일부 sin 수식 (`# + 0.5 * vx_mps * speed_scaler * np.sin(...)`) — 원본 자체가 주석.

### 검증

- `colcon build --packages-select obstacle_publisher` ✅ 0.8s
- `pytest test/test_opponent_resampling.py` ✅ **12/12** (sort, resample 선형보간/clamp, find_nearest_idx, advance_s wrap)
- **3-노드 통합** (global_republisher + frenet_conversion_server + obstacle_publisher):
  - `[ObstaclePublisher] opponent trajectory built (856 oppwpnts), start_s=10.00` ✓
  - `/tracking/obstacles` topic hz: **50.002 Hz, std 0.14ms** ✓
  - 한 메시지 디테일: `s_start=84.18, s_end=84.68` (length=0.5), `d_left=0.2, d_right=-0.2` (size=0.4), `vs=3.66 m/s`, `x_m, y_m, z_m` 모두 frenet→cartesian 변환된 정상 값
  - `/opponent_waypoints` 도 25-tick 마다 발행 ✓

### 부수 이슈 (별도 phase 로 fix 예정)

A-2b 의 `frenet_conversion_server` 가 `/global_waypoints` 가 0.5Hz 로 갱신될 때마다 새 `FrenetConverter` 인스턴스 만들면서 **track bounds 를 매번 다시 set** ("Track bounds loaded: 857 left, 857 right" 로그 반복). `_on_global_traj` 가 새 converter 빌드 시 기존 trackbounds 도 같이 옮기도록 수정 필요. **B-3 와는 별개**, 다음 작업으로 처리.

→ **fix 완료** (commit `901cf7a`): `_on_global_traj` 가 새 converter 만들 때 기존 trackbounds 옮김. 12초 검증에서 "Track bounds loaded" 로그 1회만.

---

## Phase C-2 — state_machine 순수 모듈 포팅 결과 (2026-05-04)

state_machine 의 sub-phase 첫 단계. SH 가 ROS1 단계에서 분리해 둔 순수 모듈을
ROS2 ws 로 옮기고 67 pytest 통과 확인. 메인 노드 (1212L) 와 mixin 모듈은 C-3..C-6.

### 포팅 대상 (state_machine 패키지의 lib 부분)

| 파일 | ROS2 (state_machine/) | 변경 |
|---|---|---|
| `states_types.py` (12L) | 그대로 | enum 만 |
| `states.py` (59L) | TYPE_CHECKING import 만 패키지 경로 | f110_msgs.msg.Wpnt 만 |
| `path_checker.py` (367L) | rospy.loginfo/logwarn → logging 표준 | _logger 한 인스턴스로 통일 |
| `state_transitions.py` (490L) | 동일 + 패키지 내부 import | sed 자동 변환 |
| `waypoint_data.py` (134L) | dynamic_reconfigure 제거 + ROS2 attach_to_node 패턴 stub (C-3 에서 채움) | for_test 그대로 유지 |

### **신규 패턴 — 모듈 레벨 logging** (이후 ROS-free lib 모듈에 재사용)

원본 ROS1 의 `rospy.loginfo/logwarn` 을 표준 Python `logging` 으로 대체:
```python
import logging
_logger = logging.getLogger(__name__)
...
_logger.info("...")
_logger.warning("...")
```

ROS2 launch 환경에서도 정상 작동 (`rclpy` 가 standard logging 으로 통합). 테스트에서는
mock 불필요 — sys.modules 조작 (원본 conftest 의 `setdefault("rospy", MagicMock())`)
도 전부 제거.

### conftest 단순화

원본 ROS1 conftest:
```python
sys.modules.setdefault("rospy", MagicMock())
sys.modules.setdefault("dynamic_reconfigure", MagicMock())
sys.modules.setdefault("dynamic_reconfigure.msg", MagicMock())
_SRC_DIR = ".../state_machine/src"
sys.path.insert(0, _SRC_DIR)
```

ROS2 conftest:
```python
# rospy/dynamic_reconfigure mock 불필요
# state_machine 패키지가 ament_python install 되므로 from state_machine.x 직접 import
_TEST_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _TEST_DIR)  # fake_msgs 만 위해 test 디렉터리 추가
```

### 검증

- `colcon build --packages-select state_machine` ✅ 0.8s
- `pytest test/` ✅ **67/67** (path_checker 40 + state_transitions 27)
- 단 1개 fail 후 fix: `for_test` 가 ROS2 에서 `dyn_sub` 필드 제거했는데 테스트가 검사 → legacy 호환 위해 `obj.dyn_sub = None` 1줄 유지

### waypoint_data 의 ROS2 wiring (C-3 에서 작업)

(C-3 에서는 init mixin 의 `_get_param_or_default` helper 만 추가. waypoint_data 의 attach_to_node 는 C-5 에서 ROS2 native parameter callback 으로 활성화 예정)

---

## Phase C-3 — state_machine mixin 모듈 포팅 결과 (2026-05-04)

state_machine sub-phase 두 번째. 4개 mixin 모듈 (`init` / `callbacks` / `viz` /
`smart_helper`) 의 rospy → rclpy 변환. 메인 노드 (1212L) 와 dynamic_reconfigure
서버 통합은 C-4 / C-5 영역.

### 변환 매핑 (자동화 sed + 수동 편집 조합)

자동 sed 변환 (모든 mixin 에 일관 적용):

| ROS1 (rospy) | ROS2 (rclpy) |
|---|---|
| `import rospy` | 제거 |
| `from rospkg import RosPack` | 제거 (대신 `_resolve_stack_master_path` helper) |
| `from dynamic_reconfigure.msg import Config` | 제거 (type hint 만 `Any` 로 임시) |
| `rospy.get_param(name, default)` | `self._get_param_or_default(name, default)` (helper) |
| `rospy.loginfo / logwarn / logerr / logdebug` | `self.get_logger().info / warning / error / debug` |
| `rospy.Time.now()` | `self.get_clock().now().to_msg()` |
| `rospy.Subscriber("/topic", MsgT, cb)` | `self.create_subscription(MsgT, "/topic", cb, 10)` |
| `rospy.Publisher("/topic", MsgT, queue_size=N)` | `self.create_publisher(MsgT, "/topic", N)` |
| `rospy.wait_for_message("/topic", MsgT)` | `wait_for_message(MsgT, self, "/topic", time_to_wait=10.0)` (rclpy.wait_for_message) |
| `import states` (모듈 import) | `from state_machine import states` (패키지 내부) |

수동 편집:
- multi-line `rospy.Subscriber(...)` / `rospy.Publisher(...)` (sed 단행 패턴 못 잡음)
- `RosPack().get_path('stack_master')` 두 곳 → `_resolve_stack_master_path()` helper
- 외부 의존 (vesc / dynamic_reconfigure Config sub 4개) → 주석 처리 (TODO C-5)

### 신규 helper (init mixin 에 정의)

```python
def _get_param_or_default(self, name, default=None):
    """ROS1 의 rospy.get_param 호환. declare_parameter 가 안 됐으면 default 로 declare 후 반환."""
    if not self.has_parameter(name):
        if default is not None:
            self.declare_parameter(name, default)
        else:
            return None
    return self.get_parameter(name).value

def _resolve_stack_master_path(self, *parts) -> str:
    """stack_master 경로 fallback (ROS1 ws — B-1/B-6 패턴)."""
    return os.path.expanduser(
        os.path.join("~/unicorn_ws/ICRA2026_HJ/stack_master", *parts)
    )
```

### 미포팅 / 보류 (의도)

| 항목 | 처리 | C-4/C-5 에서 |
|---|---|---|
| `vesc/sensors/core` (VescStateStamped) | 주석 처리 | 미포팅 — voltage 모니터링 비활성. 검증 필수 아님 |
| `dynamic_reconfigure.msg.Config` sub 4개 | 주석 처리 | C-5 에서 ROS2 native parameter callback 으로 통합 |
| `trajectory_planning_helpers` (tph) — ggv/ax_max 로딩 | conditional import + ImportError on use | C-4 검증 시 `pip install trajectory_planning_helpers` 결정 |

### 검증

- `colcon build --packages-select state_machine` ✅ 0.8s
- 4 mixin 모두 smoke import OK:
  - `InitMixin` (369L): _load_rosparams, _load_vehicle_dynamics, _load_vel_planner_params, _init_state_attributes, _setup_ros_subscribers, _setup_ros_publishers, _get_param_or_default, _resolve_stack_master_path
  - `CallbackMixin` (285L): 24 콜백 (avoidance_cb, dyn_param_cb, ego_prediction_cb, ...)
  - `VisualizationMixin` (224L): publish_not_ready_marker, visualize_state
  - `SmartStaticChecker` (143L): __getattr__ proxy + _odom_fixed_cb + update
- **회귀 — 기존 67 pytest 그대로 통과** ✓ (path_checker 40 + state_transitions 27)

다음 — C-4 (메인 노드 1212L 변환). C-3 의 모든 mixin 이 메인 노드의 `class StateMachine(InitMixin, VisualizationMixin, CallbackMixin)` 으로 다중 상속될 것.

---

## Phase C-4 — state_machine 메인 노드 포팅 결과 (2026-05-04)

state_machine sub-phase 세 번째. 1212L 메인 노드 변환. C-3 의 mixin 들과 결합.

### 변환 매핑

C-3 의 자동 sed 패턴 모두 재사용 (rospy.* → ROS2). 추가 변경:

| 원본 ROS1 | ROS2 |
|---|---|
| `class StateMachine(InitMixin, VisualizationMixin, CallbackMixin)` | `class StateMachine(Node, InitMixin, VisualizationMixin, CallbackMixin)` |
| `def __init__(self, name): self.name = name; self._load_rosparams(); ...` | `Node.__init__(self, name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)` 먼저 |
| `self.loop()` 무한 루프 | `self.create_timer(1/rate_hz, self.loop)` + 메인은 `rclpy.spin(node)` |
| `if __name__ == "__main__": rospy.init_node + while not is_shutdown` | `def main(args=None): rclpy.init/spin/shutdown` |
| `rospy.X_throttle(period, msg)` (멀티라인) | `self.get_logger().X(msg)` (throttle 효과는 일단 제거. 필요 시 향후 add) |

### Node.__init__ 의 중요 옵션 — `automatically_declare_parameters_from_overrides=True`

이게 없으면 launch 에서 `-p name:=value` 로 준 모든 파라미터를 코드 안에서 일일이 `declare_parameter` 해야 함. ROS1 의 `rospy.get_param` 이 declare 없이 자동이었던 동작과 비호환.

`True` 로 설정하면 launch 의 모든 파라미터가 자동 declare → `get_parameter(name).value` 그대로 사용. ROS1 → ROS2 마이그레이션 시 사실상 필수 옵션.

### `_get_param_or_default` 강화 — Deprecated 경고 제거

ROS2 Jazzy 부터 `declare_parameter("name")` (default 없음) 이 deprecated. helper 를 try/except 패턴으로 강화:

```python
def _get_param_or_default(self, name, default=None):
    try:
        return self.get_parameter(name).value
    except Exception:
        if default is not None:
            try:
                self.declare_parameter(name, default)
                return self.get_parameter(name).value
            except Exception:
                return default
        return None
```

### 미포팅 (의도, conditional import)

| 의존 | 처리 |
|---|---|
| `trajectory_planning_helpers as tph` | `try/except ImportError → tph = None`. 사용 시 ImportError 명시 |
| `vel_planner_25d.vel_planner.calc_vel_profile` | 동일 패턴 |
| `dynamic_reconfigure.msg.Config` (type hint) | `: Any` 로 변경 |

### 검증

- `colcon build --packages-select state_machine` ✅ 0.8s
- `ros2 pkg executables state_machine` → `state_machine state_machine` ✓ (entry_point 등록)
- `python3 -c "from state_machine.state_machine_node import StateMachine, main"` ✓
- **MRO 검증**: `[StateMachine, Node, InitMixin, VisualizationMixin, CallbackMixin, object]` ✓
- **회귀**: 기존 67 pytest 그대로 통과 ✓
- `ros2 run state_machine state_machine` 5초 기동:
  - Deprecated 경고 모두 제거됨 ✓
  - `_load_rosparams` 의 `self.sectors_params["n_sectors"]` 에서 `TypeError: 'NoneType'` (의도된 fail — launch 에서 `/map_params` 안 줌)
  - = startup 진입 + import + class instantiation 통과 ✓

### 다음 — C-5

`dynamic_*_server` (108 + 88L) 두 노드를 ROS2 native parameter callback 으로 통합. 메인 노드의 `add_on_set_parameters_callback` 으로 dyn_param_cb 등이 활성화됨. C-3 의 주석 처리한 4개 sub 도 동시에 ROS2 patternd 로 활성화.

`__init__` 에 `node` 파라미터 추가 (옵셔널). `attach_to_node(node)` 메서드는 stub —
C-3 에서 다음 패턴으로 채움:
```python
def attach_to_node(self, node):
    for name, default in self._param_defaults().items():
        node.declare_parameter(f"{self.name}/{name}", default)
        setattr(self, name, node.get_parameter(f"{self.name}/{name}").value)
    node.add_on_set_parameters_callback(self._on_param_change)
```
