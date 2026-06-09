# MPCC Algorithm Reference (IFAC2026_SH `nonlinear_mpc_acados`)

> 이 문서는 우리 MPCC가 **어떻게 작동하는지** 전체를 설명한다: 모델(dynamic 여부),
> 비용함수(cost), 제약(constraints), 속도 레퍼런스, LMPC, 그리고 **각 파라미터의 의미**와
> **지금 쓰는 파일들**. (버그 수정 이력은 제외 — 알고리즘 자체만.)
> 작성: 2026-06-09.

---

## 0. 한눈에

- **종류**: MPCC (Model Predictive **Contouring** Control) — 레퍼런스 라인을 따라가되
  진행(progress)을 최대화하는 racing MPC.
- **솔버**: acados **SQP_RTI + HPIPM** (실시간 1-iteration), 비용형식 **CONVEX_OVER_NONLINEAR(CONL)**.
- **모델**: ✅ **DYNAMIC 8-state 자전거 모델 + Pacejka 타이어(tanh)**. (kinematic 5-state도 코드에 있으나 폐기 — `use_dynamic: true`.)
- **호라이즌**: N=20 stage × dT=0.04s = **0.8s** (Euler 적분).
- **레퍼런스**: `track_source: raceline` → IQP 최적화 racing line(`global_waypoints.json`). corridor는 좌우 벽(d_left/d_right) 안.
- **모델오차 보상**: LMPC(on). GP residual / error-regression은 off (실차 큰 mismatch용).

---

## 1. 차량 모델 (dynamic, 8-state Pacejka)

상태 `x` (8): `[px, py, ψ, vx, vy, r, s, δ_prev]`
- `px, py` 위치, `ψ` heading, `vx` 종방향 속도, `vy` 횡방향 속도, `r` yaw rate,
  `s` 진행 arc-length, `δ_prev` 직전 조향각(steer-rate 비용용).

입력 `u` (3): `[a_x, δ, p_v]`
- `a_x` 종방향 가속도, `δ` 조향각, `p_v` progress 속도(arc-rate 변수).

동역학: 동적 자전거 모델 + 타이어 횡력 `F_y`.
- `dyn_tire_model: tanh` — `F_y = D·tanh(C·α)` 형태 (slip각 α 포화 모델링; linear/pacejka도 선택지).
- slip각: `α_f = atan2(-vy - lf·r, vx_safe) + δ`, `vx_safe = fmax(vx, dyn_v_eps)` (저속 vx→0 특이점 방지).
- `lm_dynamic` = acados Levenberg–Marquardt 정규화 (QP 조건수 ↑, MINSTEP/발산 ↓).

**joint-α LMPC 확장**: `use_lmpc=true`면 상태에 α(10개) 추가 → nx 8→18 (LMPC terminal이 α를 같이 최적화).

---

## 2. 비용함수 (CONVEX_OVER_NONLINEAR)

형식: `ψ(r) = ½·rᵀ·W·r` (잔차 `r`=`y_expr`, Gauss-Newton Hessian = NLS와 동일하되 선형항 추가 가능).

### Stage 잔차 `y_expr` (9개) 와 가중치 `W`
| # | 잔차 (residual) | 의미 | baked weight `W_def` | live scale |
|---|---|---|---|---|
| 1 | `e_c − e_c_ref` (×√att) | **contouring 오차** (라인 횡거리) − apex bias | `q_cte_def=15` | `q_cte_scale_live` |
| 2 | `e_l` (×√att) | **lag 오차** (라인 따라 진행축 오차) | `q_lag_def=80` | `q_lag_scale_live` |
| 3 | `yaw_err` | **heading 오차** | `q_psi_def=10` | `q_psi_scale_live` |
| 4 | `vx_for_cost − ref_v` | **속도 추종** (ref_v 따라가기) | `q_v_def=12` | `q_v_scale_live` |
| 5 | `δ` | **조향 크기** 페널티 (큰 조향 억제) | `q_dd_def=5` | `q_dd_scale_live` |
| 6 | `p_v − speed_target` | **progress 보상** (목표속도까지 진행) | `q_p_def` | `q_p_scale_live` |
| 7 | `side_term` | **장애물 회피 측면선호** (무장애물=0) | `q_side_def` | — |
| 8 | `δ − δ_prev` | **조향 변화율(rate)** 페널티 (튐/chatter 억제) | `q_d_rate_def` | `q_drate_scale_live` |
| 9 | `a_x` | **종방향 가속도** 페널티 (급가감속 억제) | `q_dv_def` | `q_dv_scale_live` |

- 실효 가중치 = `W_def × q_*_scale_live` (잔차에 `√(scale)` 곱해 구현).
- `e_c = sinψ_ref·(x−ref_x) − cosψ_ref·(y−ref_y)` (라인 법선방향 거리 = contouring).
- `e_l = −cosψ_ref·(x−ref_x) − sinψ_ref·(y−ref_y)` (라인 접선방향 = lag).
- `e_c_ref = −D_apex·tanh(signed_κ/0.20)` — **apex bias**: 코너 안쪽으로 목표 횡거리 이동(D_apex>0). raceline 모드선 라인 자체가 apex 통과해서 D_apex≈0 권장.
- `√att` (attenuation): 장애물 근처서 contouring 비용 감쇠(detour 허용).

### Terminal 잔차 `y_expr_e` 와 `W_e`
- dynamic: `[e_c×5, e_l×5, yaw×4, vterm×3, (LMPC residual)]`
- joint-α LMPC: `[e_c×5, e_l×5, yaw×4, vterm×3, anchor×4, cog]`
  - **anchor**: 종단 상태 `x_N`를 safe-set의 α-조합점으로 끌어당기는 soft 비용.
  - **cog (cost-to-go)**: `Qᵀα` 가 ψ_e에 **선형**으로 들어감(`lmpc_cog_w`) → Rosolia value-function 근사.
- `vterm` = 종단 vx cost-to-go (×3).

---

## 3. 제약 (constraints)

### 입력 박스 (`idxbu=[0,1,2]`)
- `a_x ∈ [a_min_dyn=−3.0, a_max_dyn=4.0]` (제동/가속 한계; −8.26 대신 −3로 줄여 vx<0 발산 방지).
- `δ ∈ [−mpc_max_steering, +mpc_max_steering]`
- `p_v ∈ [0, p_max]`

### 상태 박스 (`idxbx=[3,4,5]` = vx, vy, r)
- `vx ∈ [0, v_max+0.5]` (후진 차단 + 작은 margin; v_max 빡빡하면 IPM 발산).
- `vy ∈ [−10,10]`, `r ∈ [−20,20]`.
- joint-α면 `α ∈ [0,1]` (K=10개) 추가.
- **per-stage**: 각 stage에서 곡률 기반 `√(a_lat_safe/|κ|)`로 vx ubx를 추가로 조임 (dynamic은 ×1.6 generous — corridor cap이 안전 담당).

### 비선형 h-제약 (전부 **soft**, slack `idxsh=[0,1,2,3]`)
순서 `[h_obs, h_corridor_top, h_corridor_bot, a_lat]` (+joint면 `Σα=1` eq):
1. **h_obs**: 장애물 half-plane (회피) — `side_pref·(e_c − e_c_obs) − (R_safe+R_car) ≥ 0`.
2. **h_corridor_top/bot**: `e_c ∈ [lower_lat, upper_lat]` — 차 중심을 코리도어(좌우 벽−margin) 안에 유지.
3. **a_lat**: `|vx²·κ| ≤ a_lat_max` — 횡가속도(그립) 한계.
- slack 가중치 `zl/Zl` (corridor=20/15, obs=40/30, a_lat=50/15). **soft라서 progress 비용이 강하면 경계를 살짝 넘을 수 있음** (그래서 좁은 데서 벽클립 가능 → corridor 속도캡으로 보완).

---

## 4. 속도 레퍼런스 & corridor (track_loader.py)

`build_track_from_wpnts()` 가 wpnt별 `ref_v`와 corridor 경계 생성:
- `ref_v[i] = clip( min( vx_mps×vel_scale,  √(a_lat_safe/κ_eq),  default_v,  corridor_speed_cap ), 1.0, default_v )`
- **brake-aware κ_eq** (forward window): `κ_eq = κ'/(1+2·bf·d·κ')` (bf=0.7) — 멀리 있는 코너는 제동거리만큼 할인 → 완만구간 가속 허용, tight 코너 제때 감속.
- **corridor_speed_cap(width)**: 폭 좁은 곳 감속 (κ-cap 사각지대=좁고직선 벽클립 방지). `v_floor`(≤tight) ~ `v_full`(≥wide) 선형. (crash-fix)
- **forward-backward brake profile**: 코너 전 미리 감속 전파 (TUM calc_vel_profile 식).
- corridor 경계: 중심 ± (d_left/d_right − `_WALL_MARGIN`=0.18). `psi_centerline_rad` 기준 법선 투영 (centerline tangent일 때 정확).
- `mpc_corridor_half_width>0`이면 고정폭 corridor (벽 무시).

---

## 5. LMPC (Learning MPC, Rosolia 2018)

- `use_lmpc=true`: lap-by-lap로 **safe-set**(과거 클린 랩 궤적+cost-to-go) 학습 → terminal 비용이 거기로 끌어당김 → 모델 mismatch(gym ST vs MPCC Pacejka) 자동 보상.
- **joint-α**: 종단점을 safe-set의 convex 조합 `Σαᵢ·xᵢ`로 두고 α를 solver가 같이 최적화 (`Σα=1`, `α∈[0,1]`).
- **lap_database**: 랩 궤적 저장. **teleport(safe-reset) 발생한 랩은 거부**(`lmpc_max_resets=0`) — 박힌 궤적이 safe-set 오염 방지.
- **seed**: cold-start 시 IQP raceline을 합성 lap-0로 주입(`lmpc_seed_from_raceline`).
- `lmpc_enable_after_real_laps`: 실제 랩 N개 쌓여야 활성(synthetic seed만으론 X).

---

## 6. 모델오차(model mismatch) 보상 — 현재 상태

| 메커니즘 | 상태 | 설명 |
|---|---|---|
| **LMPC** | ✅ on | value-function 학습으로 사후 보상 (위 §5) |
| GP residual (`use_gp_residual`/`use_gp_casadi`) | off | gym−Pacejka 잔차를 f_expl에 주입 (실차 큰 mismatch용) |
| B4' error-regression (`use_error_regression`) | off | SS이웃 잔차 회귀 → velocity rows 보정. **constant-offset만 잡아 sim(작은 mismatch)선 marginal, 실차/저그립(큰 mismatch)서 유효** |

> ⚠️ sim `mu`를 낮춰 실차 저그립 매칭 시 mismatch가 커짐 → 이때 GP/error-regression이 의미있어짐.
> 단 MPC 예측모델 자체는 mu를 모름 → 예측-레벨 보정(GP) 또는 tire-grip 매칭이 정공법.

---

## 7. 파라미터 의미 사전 (`ddrx_unified_params.yaml`)

### 속도 / 조향
| param | 의미 |
|---|---|
| `max_speed` / `max_speed_p` | vx 하드 캡 (codegen ubx/ubu 절대상한) |
| `speed_target` | progress 비용(잔차 #6)의 목표속도 (hard cap과 decouple) |
| `lookahead_m` | κ 윈도 길이 (brake-aware κ_eq 전방 탐색) |
| `mpc_max_steering` | δ 박스 한계 (rad) |
| `cold_start_vx_floor` | 정지출발 시 solver x0의 vx 하한 (저속 ill-cond 회피) |
| `startup_speed` | 출발 시 출력 speed 하한 |
| `vel_scale` | raceline `vx_mps`에 곱하는 배율 |

### 호라이즌 / 모델
| param | 의미 |
|---|---|
| `N_horizon` | 예측 stage 수 (×dT = 호라이즌 시간) |
| `dT` | stage 간격 (제어주기 = 1/dT Hz) |
| `time_steps_mode` | `uniform`(전 stage dT) / `pyramidal`(multi-dt, 현재 미사용 — cost가 stage-dt 미인지) |
| `use_dynamic` | true=8-state Pacejka, false=kinematic 5-state |
| `dyn_tire_model` | `linear`/`tanh`/`pacejka` 타이어 횡력 모델 |
| `lm_dynamic` | acados LM 정규화 (QP 조건수) |
| `integration_mode` | `Euler` |
| `track_source` | `raceline`(IQP 최적선) / `centerline`(중앙선) |

### 코스트 가중치 (§2의 9잔차, `_scale_live` = baked `_def`에 곱)
| param | 영향 잔차 | 의미 |
|---|---|---|
| `q_cte_scale_live` | #1 | contouring(라인 횡추종) 강도 ↑=라인 빡빡 |
| `q_lag_scale_live` | #2 | lag(진행축) 강도 |
| `q_psi_scale_live` | #3 | heading 추종 |
| `q_v_scale_live` | #4 | 속도(ref_v) 추종 |
| `q_dd_scale_live` | #5 | 조향 크기 페널티 |
| `q_p_scale_live` | #6 | progress 보상 ↑=더 공격적 진행 |
| `q_drate_scale_live` | #8 | 조향 변화율 페널티 ↑=부드러움(튐↓) |
| `q_dv_scale_live` | #9 | 가속도 페널티 ↑=부드러운 가감속 |
| `a_lat_safe_live` | ref_v + a_lat 제약 | 코너 횡가속 한계 ↑=코너 더 빠르게(공격적) |
| `D_apex_live` | `e_c_ref` | apex bias 깊이 (코너 안쪽 컷) |
| `alpha_steer_live` | — | 조향 출력 1차 필터 계수 |
| `cost_spike_thr_live` | — | 비용 스파이크 fallback 임계 |
| `commit_dist_live` | — | 장애물 회피 commit 거리 |

### corridor 속도캡 (crash-fix)
| param | 의미 |
|---|---|
| `corridor_v_floor` | 좁은 통로 최저 속도 (0=캡 끔) |
| `corridor_v_tight_width` | 이 폭(post-margin) 이하 → v_floor |
| `corridor_v_wide_width` | 이 폭 이상 → 감속 없음(full) |
| `mpc_corridor_half_width` | >0이면 고정폭 corridor(벽 무시), 0=실제 d_left/d_right |

### LMPC
| param | 의미 |
|---|---|
| `use_lmpc` | LMPC terminal 활성 |
| `lmpc_w` | LMPC terminal 잔차 가중 |
| `lmpc_alpha` / `lmpc_beta` | safe-set 거리 페널티 / softmax 날카로움 |
| `lmpc_reg_w` | α 정규화 (Hessian 양정칙) |
| `lmpc_K_points` | safe-set kNN 개수 (=10, codegen 고정) |
| `lmpc_slice_window` | 현재 s 근처 ± step 윈도 |
| `lmpc_max_resets` | 랩당 허용 teleport 수(초과=safe-set 거부; 0=teleport랩 전부 거부) |
| `lmpc_max_abs_ec_m` / `lmpc_max_lap_time_ratio` / `lmpc_max_stuck_seconds` | 랩 수락 필터(corridor이탈/느린랩/stuck) |
| `lmpc_buffer_per_bucket` | v_bucket별 보관 랩 수 |
| `lmpc_seed_from_raceline` | cold-start raceline seed |
| `lmpc_enable_after_real_laps` | 실제 랩 N개 후 활성 |

### 모델오차 / 기타
| param | 의미 |
|---|---|
| `use_gp_residual` / `gp_ckpt_path` | torch GP residual wrap |
| `use_gp_casadi` | closed-form GP를 dynamics에 baking |
| `use_error_regression` / `err_regr_bandwidth` | B4' 잔차회귀 + Epanechnikov 대역폭 |
| `auto_tune` | true면 max_speed 하나로 나머지 자동매핑 (false=yaml값 그대로) |
| `use_ml_scale` / `ml_model_path` | MLP weight scaler |
| `override_mode` | `'off'` (legacy bucketed/poly 비활성) |
| `auto_step_*` | 같은 launch서 lap기반 v_max 자동 증가(현재 off) |

### sim 마찰 (`stack_master/config/SIM/sim_params.yaml`)
| param | 의미 |
|---|---|
| `mu` | gym 노면 마찰계수 (기본 1.0489; 실차 저그립 매칭 시 낮춤, 예 0.6) |

---

## 8. 지금 쓰는 파일들

### 코드 (런타임)
- `src/nonlinear_mpc_acados/nonlinear_mpc_acados/`
  - `mpc_node.py` — ROS2 노드: 제어 틱, solve 호출, odom, LMPC 랩 버퍼/쿼리, 명령 publish, stuck 복구.
  - `track_loader.py` — 트랙/raceline 로드, corridor 경계, ref_v(brake-aware κ + corridor cap + brake profile).
  - `mpc_core/acados_kinematic.py` — **MPCC 핵심**: 모델 build, CONL cost, 제약, solve. (acados 백엔드)
  - `mpc_core/ipopt_kinematic.py` — IPOPT 백엔드(대안).
  - `mpc_core/lmpc/` — `lap_database.py`, `safe_set.py`, `nominal_dynamics.py`, `error_regression.py`.
  - `mpc_core/refv_smoothing.py` — ref_v κ-step 평활(모듈, tight맵선 no-op).
  - `pp_fallback_node.py`, `ftg_fallback_node.py` — MPCC 죽으면 mux가 쓰는 폴백.
  - `ml/` — MLP weight scaler (model/inference/train).

### 설정
- `src/nonlinear_mpc_acados/config/ddrx_unified_params.yaml` — **메인 파라미터** (§7).
- `src/nonlinear_mpc_acados/config/mpc/BO_params_*.json` — BO 튜닝 weight 프리셋.
- `src/nonlinear_mpc_acados/config/tire/{linear,pacejka}.yaml` — 타이어 파라미터.
- `src/stack_master/config/SIM/sim_params.yaml` — gym `mu` 등 sim 물리.
- `src/stack_master/config/SIM/sim.yaml` — sim 토픽/노드 설정.

### 맵 (`src/stack_master/maps/<map>/`)
- `global_waypoints.json` — **핵심**: centerline_waypoints + global_traj_wpnts_iqp(raceline). 각 wpnt: `x_m,y_m,psi_rad,d_left,d_right,kappa_radpm,vx_mps,...`
- `<map>.png` + `<map>.yaml` — occupancy(코리도어/충돌). yaml `image:`는 png 가리켜야 함.
- `start_pose.yaml` — 차 스폰 포즈(raceline[0] 기준). `ot_sectors.yaml`, `speed_scaling.yaml` — 섹터(현재 빈 리스트).
- 현재 맵: `final`, `final2`.

### 런치
- 시뮬: `ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=<map>` (gym + mpc_node + mux + rviz 한방).
- (실차 ws `nonlinear_mpcc`는 2-launch: `stack_master low_level.launch.xml sim:=true` + `nonlinear_mpcc mpcc.launch.xml`.)
- acados codegen: 첫 launch ~수십초. 모델/제약/N/time_steps 바꾸면 `rm -rf /tmp/acados_codegen_evompcc` 후 재생성.

### BO (파라미터 최적화)
- `scripts/bo_sweep_turbo.py` (BoTorch TuRBO + ConstrainedEI, 9D: q_cte/q_lag/q_psi/q_v/q_p/q_drate/q_dv/a_lat_safe/D_apex), `scripts/run_bo.sh` (clobber-safe: 종료 시 yaml 복원, best는 `~/bo_results/bo_turbo_*.json`의 `raw_best_params` 사용).
- reward: `scripts/eval_run_quality.py` (랩타임 + stuck/shake 페널티).

---

## 9. 현재 deploy 기본 동작 요약

1. global_waypoints.json 로드 → raceline 레퍼런스 + corridor 경계 + ref_v(brake-aware κ + corridor cap).
2. 매 제어틱(25Hz): 현재 상태 → acados SQP_RTI solve (dynamic 8-state, CONL cost, soft corridor/a_lat 제약) → u[0]의 제어를 mux 통해 차에 publish.
3. LMPC: 매 랩 클린 궤적을 safe-set에 누적(teleport랩 거부) → terminal 비용이 학습된 라인으로 끌어당김.
4. stuck 시: 후진 복구 → 안되면 /sim/initialpose teleport(sim) / 경고(실차).
5. MPCC 죽으면 mux가 pp_fallback으로 자동 전환.
