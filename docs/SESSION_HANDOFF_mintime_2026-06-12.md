# Session Handoff — mintime raceline A/B (2026-06-12)

노트북에서 이어서 작업하기 위한 핸드오프. 브랜치: `avoidance-restore`.

## 한 줄 요약

**mu=0.6 정직 mintime 라인(final2_mt)은 안정적(STUCK 0)이지만 구 mincurv 라인보다 ~4초 느림 (24.1s vs 20.2s). 라인 교체는 보류, 원인 분석 완료.**

## A/B 결과

| | 구 라인 (final2, mincurv_iqp, ggv=12 잘못된 grip) | 신 라인 (final2_mt, mintime, mu0.6 정직) |
|---|---|---|
| 랩타임 | **20.16–20.24s** (28랩) | 23.56–24.44s, mean **24.14s** (41랩) |
| STUCK | 0 | 0 |
| 경로 길이 | 74.94 m | 71.73 m |
| 로그 | mpc_logs/mpc_20260612_144857.csv | mpc_logs/mpc_20260612_174736.csv |

공통 조건: sim mu=0.6, dyn_mu:=0.6, LMPC off, mode:=mpcc, a_lat clamp 5.59 (μgη).

## 왜 mintime이 졌나 (진단)

1. **κ-스파이크 2곳** (s≈40, s≈71, R≈0.45m): reopt 후에도 잔존. ref_v의 κ-cap
   `√(a_lat_eff/κ)` 이 그 지점에서 1.6–1.9 m/s 크롤 강제. s-bin 속도 분석에서
   s=14–18 / 38–42 / 54–58 / 70–72 구간 v_min 1.5–1.9 m/s 확인.
2. **가속 가정 불일치**: mintime은 f_drive_max=33.4N (≈9.5 m/s² 저속 가속)을 가정해
   "느린 코너 → 폭발 가속" 라인을 그렸지만 우리 컨트롤러/플랜트는 a_long ≈3 m/s²
   수준 — 코너 탈출 가속을 못 살림.
3. 경로 3.2m 단축으로는 위 손실 보상 불가.

구 라인이 빠른 이유: mincurv는 곡률을 globally 부드럽게 깔아서 κ-cap ref_v가
크롤 지점 없이 흐름 — **"틀린 ggv로 만든 라인"이지만 geometry 자체는 우리
컨트롤러의 실제 가속 한계와 우연히 더 잘 맞음.**

## 다음 후보 (우선순위 제안)

1. **mintime을 컨트롤러-정직 가속으로 재실행**: racecar_f110.ini 의
   f_drive_max/f_brake_max 를 a_long 3–4 m/s² 상당(≈10.5–14N)으로 낮춰 재최적화
   → "가속 못 살리는 라인" 문제 직접 해소. 스크립트 그대로 재사용 가능.
2. κ-스파이크 2곳 post-smooth (s≈40, 71) 후 재시도.
3. 라인 교체 포기하고 구 라인 + BO 재튜닝 (20.2s 베이스라인에서).

## 재현 방법 (이 repo만으로)

```bash
# 빌드 (final2_mt 맵 포함)
colcon build --packages-select stack_master --symlink-install
# A/B 주행
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final2_mt dyn_mu:=0.6   # 신 라인
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final2    dyn_mu:=0.6   # 구 라인
```

헬스체크: mpc_node 로그에 `N_orig=719, L_orig=71.73` (final2_mt) / clamp 5.59 경고.

## mintime 재생성 (라인 다시 뽑을 때)

`scripts/mintime_final2/` 에 전체 파이프라인 보존:

- `run_mintime.py` — 드라이버. **주의: 절대경로 의존** (이 머신 기준):
  - GRO 원본: `/home/hmcl/creating_autonomous_car_ws/src/creating_autonomous_car/planner`
    (TUMFTM global_racetrajectory_optimization — 노트북에선 경로 수정 필요)
  - spliner/gb_optimizer_25d/vel_planner_25d install site-packages (이 repo colcon build 산출물)
- `traj_opt_patched.py` — gb_optimizer_25d trajectory_optimizer 사본,
  `reopt_mintime_solution=True` + `recalc_vel_profile_by_tph=True` (κ≤curvlim 강제 + mu0.6 ggv 속도 재계산)
- `opt_mintime_patched.py` — GRO opt_mintime 사본, casadi DM→float() 패치 4줄 (566/567/679/680)
- `racecar_f110.ini` — v_max 8.0, mue 0.6 으로 수정된 버전
- `veh_dyn_info/ggv.csv` — 전 속도 5.886/5.886 (mu0.6·g)
- `mintime_traj.npz` — 이번 결과 (719pts, L=71.73m, est 16.59s)
- `make_map.py` — npz → final2_mt 맵 디렉토리 설치 (wpnts/markers/csv/png/yaml,
  psi_centerline_rad 포함 — race_stack 규약 그대로)

알려진 함정: scipy euclidean 1-D 패치 필요(run_mintime.py 안에 있음),
map_info_str 은 {'data': str} dict, full_sim 은 `mode:=mpcc` 필수 (기본 timetrial 은 mpc_node 안 띄움).

## 미해결 백로그 (이전 세션에서 이월)

- LMPC per-cycle query-anchor 피드백 수정 (use_lmpc off 상태, 3691fdb)
- BO 재가동 (20.2s 베이스라인 기준)
- 실차 포팅: mpc_max_steering 0.39, VESC speed-preview kp 재계산
