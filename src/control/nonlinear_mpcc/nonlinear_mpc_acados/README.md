# nonlinear_mpc_acados

ROS2 Jazzy 용 **EVO-MPCC kinematic acados** 솔버. unicorn-racing-stack 의
ROS1 노드를 IFAC 2026 데모용으로 포팅. race-stack 의 `/centerline_waypoints`
또는 `/global_waypoints` (IQP raceline) 를 mpc reference 로 사용, fixed-width
corridor 안에서 acados SQP_RTI 로 매 cycle ~3ms 에 N=18 stage 예측.

```
              ┌─────────────────────┐
  /car_state ─┤                     ├─→ /vesc/high_level/ackermann_cmd
              │      mpc_node       │   (cmd: speed, steering)
              │   (rclpy wrapper)   ├─→ /mpc_trajectory (Path)
              │                     │   /mpc_trajectory/markers (MarkerArray)
              │  ┌───────────────┐  ├─→ /mpc/cost, solve_time, is_feasible
              │  │  mpc_core/    │  ├─→ /mpc_debug (16-field MultiArray)
              │  │  acados_      │  ├─→ /center_path /right_path /left_path
              │  │  kinematic.py │  │   /reference_path (MarkerArray, latched)
              │  └───────────────┘  ├─→ /boundary_marker (MarkerArray)
              └─────────┬───────────┘
                        │
  /centerline_waypoints ┘   ← reference (track_source: centerline)
  /global_waypoints     ┘   ← reference (track_source: raceline)
```

---

## 의존성

### 시스템 (apt 또는 수동 설치)
- **ROS2 Jazzy** (`ros-jazzy-desktop`, `ros-jazzy-joy`)
- **acados** (~/acados 에 설치) — https://docs.acados.org/installation/
- **t_renderer** 바이너리 (~/acados/bin/t_renderer)

### Python (pip)
- `casadi`, `numpy`, `scipy`
- `~/acados/interfaces/acados_template` (acados Python bindings, `pip install -e`)

### ROS2 패키지 (워크스페이스 내)
- `osuf1_common` (MPCTrajectory msg)
- `f110_msgs` (LapData, WpntArray)
- `ackermann_msgs`

---

## 설치 (fresh 시스템)

```bash
# 1. acados
git clone https://github.com/acados/acados.git ~/acados
cd ~/acados && mkdir build && cd build
cmake -DACADOS_WITH_QPOASES=ON .. && make install
pip install -e ~/acados/interfaces/acados_template

# 2. t_renderer
mkdir -p ~/acados/bin
wget -O ~/acados/bin/t_renderer \
  https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux
chmod +x ~/acados/bin/t_renderer

# 3. Python deps
pip install casadi numpy scipy

# 4. 워크스페이스 빌드
cd ~/IFAC2026_SH
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
export ACADOS_SOURCE_DIR=~/acados
```

같은 시스템에서 재빌드만 필요한 경우 4번만 수행.

---

## 실행

### Sim (gym_bridge) 통합 launch

```bash
ros2 launch stack_master full_sim.launch.py mode:=mpcc
```

- 자동으로 mpc_node + joy_node + mpc_debug_logger + sim 인프라 기동
- 약 40초 후 auto-engage helper 가 fake joy RB 1회 발행 → simple_mux 의
  autodrive 래치 ON → 차 출발
- 수동으로 즉시 출발: `ros2 topic pub --once /joy sensor_msgs/msg/Joy "{axes: [0,0,0,0,0,0], buttons: [0,0,0,0,0,1,0,0]}"`
- **첫 실행만 acados solver 코드젠 ~30초**. 이후 캐시되어 즉시 실행.

### mpc_node 단독 (별도 sim/실차 환경)

```bash
ros2 launch nonlinear_mpc_acados mpc.launch.py
```

---

## 주요 파라미터 (`config/ddrx_unified_params.yaml`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `mpc_backend` | `acados` | `acados` (SQP_RTI ~3ms) or `ipopt` (~50ms) |
| `track_source` | `raceline` | `centerline` or `raceline` (/global_waypoints) |
| `mpc_corridor_half_width` | 0.7 | corridor 좌우 폭 [m]. fixed (좌우 대칭) |
| `max_speed` | 4.0 | mpc cmd 의 절대 속도 cap [m/s] |
| `mpc_max_steering` | 0.3 | steering rad cap. 0.3 (over-shoot 방지) |
| `N_horizon` | 18 | mpc 예측 stage 수 |
| `dT` | 0.025 | 한 stage 시간 [s] (= 40Hz × 18 stage = 0.45s 예측) |
| `params_file` | `BO_params_LTM` | cost weight JSON (`config/mpc/`) |

### Live-tunable (rqt_reconfigure 또는 `ros2 param set` 으로 즉시 적용)

| 파라미터 | 기본값 | 효과 |
|---|---|---|
| `q_cte_live` | 4.0 | contour error (centerline 추종 강도). 작을수록 corridor 안 자유 |
| `q_lag_live` | 200.0 | progress (lag 비용). 크면 시간 우선 |
| `q_d_delta_live` | 25.0 | steering rate change cost (cycle 간 떨림 억제) |
| `alpha_steer_live` | 0.6 | output steering EMA (작을수록 smooth) |
| `R_safe_live` | 0.35 | 장애물 회피 측면 안전거리 |
| `R_car_live` | 0.15 | 차 반지름 (F1TENTH width 0.30 ÷ 2) |
| `D_apex_live` | 0.0 | 코너 inside apex pull bias. raceline 모드엔 0, centerline 모드엔 0.35 |
| `D_detour_live` | 0.3 | 회피 lateral offset |
| `commit_dist_live` | 15.0 | 회피 commit 시작 거리 |
| `cost_spike_thr_live` | 200.0 | cost > 이 값이면 safe fallback (ref_v·0.3) |

---

## 디버깅

### Live 로그
콘솔에 1Hz `[dbg]` 한 줄 요약 (`mpc_debug_logger` 출력):
```
[dbg] lap=0 s= 12.34 v=3.20 vcmd=3.21 steer=+0.018 solve=3.8ms cost=0.42 feas=Y margin=0.45m
```

### CSV 로그 (`~/mpc_logs/`)
- `mpc_<timestamp>.csv` — 매 cycle 전체 row (16 dbg field + meta)
- `events/event_<HHMMSS>_<reason>_<NNN>.csv` — 이상 감지 시 자동 dump
  (직전 20 cycle + 직후 15 cycle). trigger: `infeasible`, `cost_spike`,
  `vcmd_jerk`, `stuck`, `slow_solve`

### RViz 시각화 (mpcc 모드에서 자동 표시)
- `MPC_centerPath` (흰 sphere) — 트랙 centerline
- `MPC_referencePath` (노랑) — mpc 가 추종하는 reference
- `MPC_rightPath` / `MPC_leftPath` (빨강 / 초록) — corridor 경계
- `MPC_trajectory` (마젠타 sphere, 매 cycle 갱신) — mpc 예측 N stage
- `MPC_boundary` (마커 점) — inflated corridor

---

## 디렉토리 레이아웃

```
nonlinear_mpc_acados/
├── nonlinear_mpc_acados/
│   ├── mpc_node.py              ← rclpy wrapper (track 구독, control loop, viz publish)
│   ├── mpc_debug_logger.py      ← /mpc_debug → CSV + event dump
│   ├── track_loader.py          ← wpnt → CasADi spline + raw lane 저장
│   └── mpc_core/
│       ├── acados_kinematic.py  ← 1745줄, 5-state kinematic + 8-state Pacejka 토글
│       ├── ipopt_kinematic.py   ← CasADi/IPOPT 백엔드 (sim 검증용)
│       └── _ros_compat.py       ← rospy ↔ rclpy 어댑터
├── config/
│   ├── ddrx_unified_params.yaml ← 모든 ROS2 파라미터
│   ├── mpc/BO_params_*.json     ← Bayesian-Opt 결과 cost weights
│   └── tire/                    ← Pacejka / linear tire models (dynamic 용)
├── share/tracks/                ← bundled CSV 트랙 (fallback 용; raceline 모드엔 미사용)
├── launch/mpc.launch.py         ← 단독 mpc_node 실행
└── scripts/install_acados.sh    ← acados 자동 설치 헬퍼
```

---

## 참고

- 알고리즘: Liniger et al. (ETH MPCC), Kloeser et al. (IFAC 2020), Heilmeier et al. (TUM minimum-curvature)
- ROS1 원조: `unicorn-racing-stack/evo_mpcc/nonlinear_mpc_casadi/`
- BO weight 튜닝: EVO-MPCC 의 Bayesian-Opt 결과 그대로 사용

격리 원칙: 본 패키지는 race-stack 의 다른 패키지를 수정하지 않음.
유일하게 건드린 파일은 `stack_master/launch/full_sim.launch.py` (mpcc 모드
분기 추가) 와 `stack_master/config/SIM/sim.rviz` (mpc viz display 추가).
