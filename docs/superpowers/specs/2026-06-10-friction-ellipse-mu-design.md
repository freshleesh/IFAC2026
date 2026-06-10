# 설계: mu 정직화 + friction ellipse 제약 (2026-06-10)

## 배경 / 문제

sim을 mu=0.6으로 내렸을 때 (`src/stack_master/config/SIM/sim_params.yaml:1`)
MPC 내부 모델은 `dyn_mu = 1.0489` 하드코딩 상태로 남아 있었다
(`mpc_core/acados_kinematic.py:104`, config 미노출). 컨트롤러가 실제보다
~75% 더 많은 그립을 가정하고 계획 → 저그립에서 미끄러짐, mu=0.6 BO-best
재현 실패의 구조적 원인.

추가로, tanh 타이어 모델(`F_y = μ·D·F_z·tanh(B·α)`)은 횡력 포화는
모델링하지만 **종-횡 결합이 없다**: 제동 중 횡력 용량이 줄어드는 물리가
없어 "풀브레이킹 + 풀코너링"을 동시에 계획할 수 있다. 저그립일수록 이
구멍이 커진다 (참조기법 #2, friction-ellipse/slip-cap).

## 결정 (접근법 A)

mu를 파라미터로 정직화하고, 가속도공간 friction ellipse soft h-제약을
1행 추가한다. force-공간 per-axle ellipse(하중이동 포함)는 SQP_RTI
1-iteration 안정성 리스크(full Pacejka 봉인 전례)로 이번 사이클 제외.
GP residual 연결도 다음 사이클 (물리 정직화 후 남는 잔차만 GP가 담당).

## 1. 파라미터 흐름 — `dyn_mu` 노출

- `acados_kinematic.py`: 기본값 `1.0489` 유지, 기존 패턴으로 주입 추가
  (line 296~313 블록): `self.dyn_mu = param.get('dyn_mu', self.dyn_mu)`
- `mpc_node.py`: `declare_parameter('dyn_mu', 1.0489)` (use_dynamic 등
  선언부 옆, ~line 234) + `_build_param_dict()`(line 532)에 전달
- `config/ddrx_unified_params.yaml` + launch 인자로 노출 — mu=0.6 실험은
  launch 인자 하나로 전환
- **codegen-time 상수** (a_lat_safe_live와 동일 취급; BO/sweep은 노드
  재시작이므로 런타임 p_sym 불필요)

효과: tanh 타이어가 즉시 정직해짐 — 예측 모델이 저그립 한계를 알게 됨.

## 2. Friction ellipse h-제약 (1행)

식:

    (a_x / a_lim)² + (a_lat / a_lim)² ≤ 1,   a_lim = μ·g·η

- `a_lat = vx·r` (기존 표현 재사용, line 576), `a_x = u[0]`, g=9.81
- `η` = ellipse headroom 파라미터, 기본 0.95 (`ellipse_frac`)
- h 순서: `[h_obs, corr_top, corr_bot, a_lat, ellipse, (Σα if joint)]`
  — ellipse는 a_lat 행 다음, Σα(비slack 등식) 앞에 삽입
- `idxsh = [0,1,2,3,4]`, slack 가중치는 a_lat과 동일 (zl=50, Zl=15)
- 기존 flat a_lat backstop(`max(8, a_lat_safe+1)`)은 그대로 유지

회귀 안전성: mu=1.0489에서 `μgη ≈ 9.8` → ellipse가 거의 binding 안 됨
(기존 고그립 동작 보존). mu=0.6에서 `μgη ≈ 5.6` → 실질 한계. 특히
|a_x|가 클 때 허용 a_lat이 자동 감소 = combined-slip 효과.

비고: a_x 입력 박스(±3/4)는 μgη보다 작으므로 ellipse의 a_x 항은 단독
으로는 안 물리고, 가·감속 중 코너링 결합 시에만 a_lat을 깎는다 — 의도된
동작.

## 3. ref_v 프로필 μ-인지 (clamp 1곳)

`a_lat_safe_live > μ·g·η`이면 clamp + 경고 로그. 헬퍼 함수 1개
(`model_policy.py`에 배치, 두 파일에서 호출)로 중앙화. 적용 지점 3곳:

1. per-stage vx cap (`acados_kinematic.py:1878`, `√(a_lat_safe/κ)`)
2. `ocp.parameter_values[10]` (A_LAT_SAFE, line 1369/1925)
3. `build_track_from_wpnts(a_lat_max=...)` (`mpc_node.py:1034`;
   track_loader.py:277의 a_long g-g proxy에도 자동 반영)

효과: BO가 물리 한계 이상의 a_lat을 요청해도 속도 프로필이 거짓 그립
위에 세워지지 않음. BO a_lat 탐색 범위는 변경 없음 (상한이 μ에 자연히
묶임).

## 4. 검증 계획 (성공 기준: 랩타임 개선까지)

| 게이트 | 조건 | 통과 기준 |
|---|---|---|
| 유닛 | ellipse 식 + clamp 헬퍼 + codegen | 테스트 통과 (`test/` 기존 패턴) |
| 회귀 | mu=1.0489, final 맵, 3런 | 랩타임 기존 대비 노이즈 범위 내, STUCK 0 |
| 저그립 | mu=0.6, final2 맵, 3런 | STUCK 0 · 접촉 0 · 미끄러짐 없음 |
| BO | mu=0.6 BO 재가동 → best 3런 재검증 | BO-best 재현 + 랩타임 개선 |

(BO-best는 적용 전 3런 검증 필수 — 2026-06-09 교훈.)

## 범위 제외

GP residual 연결(CasADi 경로), force-공간 per-axle ellipse, 종가속
하중이동, vy/r 상태추정 필터, 장애물 의사결정 — 전부 다음 사이클.

## 영향 파일

- `nonlinear_mpc_acados/mpc_core/acados_kinematic.py` — dyn_mu 주입,
  ellipse h-행, slack 배열 확장, clamp 적용
- `nonlinear_mpc_acados/mpc_node.py` — param 선언/전달, track 빌드 시
  clamp 적용
- `nonlinear_mpc_acados/mpc_core/model_policy.py` — clamp 헬퍼
- `config/ddrx_unified_params.yaml`, launch 파일 — dyn_mu·ellipse_frac 노출
- `test/` — 유닛 테스트 추가
