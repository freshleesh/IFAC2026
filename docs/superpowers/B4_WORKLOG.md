# B4' Error Dynamics Regression — 라이브 워크로그

> **이 파일 = "지금 뭐 하고 있나"를 보는 곳.** 작업이 진행될 때마다 갱신됨.
> 같이 보면 좋은 파일:
> - 설계(왜/무엇): `docs/superpowers/specs/2026-06-04-mpcc-external-cost-rebuild-design.md` (맨 아래 "B4' 정제" 섹션)
> - 실행계획(어떻게, 파일별 edit): `docs/superpowers/plans/2026-06-05-b4-error-regression.md`
> - 이 워크로그(현재 상태): 바로 이 파일

마지막 갱신: 2026-06-05. **★ 코드 Task 1~7 전부 ✅ (구현+리뷰+커밋 완료).** 남은 것 = Task 8 = 라이브 sim 검증(메인이 sim 하나씩). e_corr↔GP 이중보정 가드 적용됨(Task 7).

---

## 1. B4'가 하는 일 (알고리즘 한눈에)

**문제:** 컨트롤러의 nominal 차량모델(8-state Pacejka tanh)은 실제 동역학과 다름(sim2real gap). 고속 한계영역에서 grip을 과신 → 미끄러지거나 접촉.

**해법 (Xue+2024 "Error Dynamics Regression"을 우리 비선형 acados에 적응):**
nominal은 그대로 두고, **속도 상태(vx, vy, r)의 오차만** 온라인으로 국소 회귀해서 보정.

```
실제 다음상태  x_{t+1}            (시뮬/실차에서 관측)
nominal 예측   x̂_{t+1} = x_t + dt·f_expl(x_t, u_t)
잔차            e_t = (x_{t+1} − x̂_{t+1})  의 [vx,vy,r] 성분      ← 랩마다 저장

매 제어주기:
  현재 상태 근처의 Safe-Set 이웃 M개의 잔차 e_i 를 가져와
  e_corr = Σ w_i·e_i / Σ w_i      (w_i = Epanechnikov 커널, bandwidth h)
  → f_expl 의 [vx,vy,r] 행에 e_corr 를 상수로 더함 (affine, horizon 전체 동일)

데이터 부족(이웃 < M_min) → e_corr = 0 (nominal 폴백, 안전)
크기 clamp → 폭주 방지
```

**핵심 설계 결정 (이번 세션 brainstorming):**
1. **affine 상수 e_corr 먼저** (per-stage 보정은 예측오차가 부족함을 *증명*할 때만 — YAGNI).
2. **검증을 위한 known-mismatch 주입**: gym의 *실제* 타이어마찰을 알려진 배율 `gym_mu_scale`로 어긋나게 → ground truth 확보. (sim 자체 mismatch는 너무 작아 lap time으로 안 보임.)
3. **정확성 게이트**: lap time이 아니라 **N-step 예측오차**(보정 < nominal)로 "작동 여부" 판정 — 맵 독립.
4. **use_lmpc=true와 결합**: 이웃 query가 LMPC Safe-Set 인프라를 재사용. use_lmpc=false면 e_corr=0 (no-op).

---

## 2. 작업 방식 (어떻게 굴러가나)

- **subagent-driven**: 태스크마다 새 subagent가 코드+pytest+colcon build → spec 리뷰 → 코드품질 리뷰 → 통과 시 커밋.
- **branch**: `lmpc-joint-alpha` (worktree 안 씀 — ROS install/ 빌드가 working tree에 있어서).
- **검증 분리**: subagent는 **코드+단위테스트+빌드**(결정적)만. **라이브 sim 측정**(acados 빌드+ROS launch+~210s lap/접촉)은 메인이 sim 하나씩(메모리 규칙) 직접 — subagent가 lap time을 못 잰다.

---

## 3. 8-태스크 진행상황

| # | 태스크 | 상태 | 커밋 | 검증 |
|---|--------|------|------|------|
| 1 | acados `f_expl` e_corr hook (p_sym 76→79, gated) | ✅ 완료 | `db18dc7` + `edb4795`(cleanup) | 코드+빌드 ✓. **sim 게이트테스트(e_corr=0→21.2s/0접촉 재현) = 보류(메인 sim 단계)** |
| 2 | `gym_mu_scale` known-mismatch 노브 | ✅ 완료 | `030f447` | 코드+빌드 ✓. **sim 로그체크 보류** |
| 3 | `nominal_dynamics.py` 공유 1-step 잔차 (pure-python TDD) | ✅ 완료 | `26429e4` | pytest 2 pass, 상수 bit-for-bit 검증 |
| 4 | per-cycle 예측오차 로거 (정확성 게이트) | ✅ 완료 | `33dae85` | 코드+빌드 ✓ (★타이밍 수정: solve 후 호출, state8↔u_seq[0] 정확 페어링). **sim 로그체크 보류** |
| 5 | `lap_database` 잔차 저장 (TDD) | ✅ 완료 | `a206ab2` + `1e70f9a`(입력로깅 fix) | pytest 7 pass |
| 6 | `safe_set` query가 이웃 잔차 반환 (TDD) | ✅ 완료 | `303cf7e` | pytest 8 pass, order 정렬 검증 |
| 7 | Epanechnikov e_corr 회귀 + mpc 배선 (+ e_corr↔GP 상호배제 가드) | ✅ 완료 | `ea23ca0` | pytest 11 pass. _err_regr=use_err∧use_lmpc, 4개 bail 모두 zero, GP guard 검증 |
| 8 | 폐루프 검증 (메인이 sim 직접) | ⏳ 다음 (sim 단계) | — | sim 측정 |

상태표는 ROS task tracker와 동기(이 대화의 TaskList).

---

## 4. 보류된 sim 검증 (Task 8에서 메인이 한꺼번에, sim 하나씩)

실행 entrypoint: `ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final use_lmpc:=true ...`
(codegen 바꾸면 `rm -rf /tmp/acados_codegen_evompcc` 먼저)

- [ ] **T1 게이트**: `_err_regr=True`·e_corr=0 → median ≈ 21.2s / 0접촉 재현 (baseline-neutral 확인)
- [ ] **T2 로그**: `gym_mu_scale:=0.9` → `[B4' mismatch] gym mu ... -> ...` 한 줄; `:=1.0`이면 무출력
- [ ] **T4 게이트**: `[B4'-pred] mean|err| nominal=.. corrected=..` 출력 (e_corr=0이면 둘 동일 — 메트릭 동작 확인)
- [ ] **T7 회복**: `use_error_regression:=true gym_mu_scale:=0.9` → lap2+ 에서 corrected < nominal
- [ ] **T8 폐루프**: 같은 mismatch서 보정 ON vs OFF → 접촉↓; a_lat↑→접촉↓; `gym_mu_scale:=1.0`서 do-no-harm

---

## 5. 결정 로그 (왜 이렇게 했나)

- **2026-06-05 commit-hygiene**: Task 2가 사용자의 기존 uncommitted launch 작업(gym_bridge_launch.py ~70줄 리팩토링, low_level 2줄)을 휩쓸어 커밋함 → 사용자 선택대로 **clean split**: 우리 커밋(`030f447`)엔 gym_mu_scale만, 사용자 기존 작업은 uncommitted WIP로 복원. gym_bridge_launch.py의 gym_mu_scale hop은 사용자 리팩토링과 얽혀있어 그 WIP에 같이 둠(working tree엔 다 있어 sim은 정상 동작).
- **Task 1 cleanup(`edb4795`)**: e_corr 3슬롯을 `n_p_stage`(per-stage 의미)에서 `n_p_const`로 이동 — e_corr은 horizon 상수라 의미상 맞음. `n_p_total`=79 불변.
- **Task 4 타이밍 수정(plan 대비)**: plan은 로거를 `_lmpc_update_per_cycle`(solve 前 호출, line 1485)에 두고 `_last_u_applied` 사용 → 적용제어가 1-cycle 어긋남(게이트 오염). 수정: state8를 그 메서드서 stash, **solve 後**에 `u_seq[0]`와 함께 로거 호출 → (state_t, u_t) 정확 페어링.
- **★ Task 5 선결버그 발견·수정(`1e70f9a`)**: 랩버퍼 `buf['input']`이 reset만 되고 append 전무 → `_lmpc_on_lap_end`가 `inputs=zeros((T-1,2))` stub 사용. 이러면 잔차 = actual − f_expl(state, **u=0**) → 모델오차가 아니라 **제어효과 전체를 흡수** = B4' 무의미 + 언패킹 크래시. 수정: post-solve서 `u_seq[0]`(3-vec [a_x,delta,p_v]) 매 cycle 로깅(use_lmpc gate, state append와 lockstep), `_lmpc_on_lap_end`서 실제 입력으로 빌드. **한계**: solve 실패가 lap 중간에 나면 state/input 1-step desync(현 truncation은 trailing만 보정) — solve 실패는 드물고 그런 lap은 보통 필터됨. 필요시 solve-fail시 state pop으로 완전 lockstep 가능(미적용, gold-plating).
- **★ 사용자 지적 — e_corr ↔ GP residual 이중보정**: 둘 다 f_expl 속도행 [vx,vy,r]에 더함 (GP `acados:636` gate=use_gp_casadi, e_corr `acados:1080` gate=_err_regr). 동시 ON이면 같은 sim2real 갭 2회 교정. 설계상 **대안**(B4'가 GP 후계자). → **Task 7서 상호배제 가드**: use_error_regression이면 GP 비활성+warn. (현재 둘 다 기본 off, GP는 실차전용이라 실제충돌 無이나 가드 필요.)

---

## 6. 안전/주의

- sim 1개만 동시에. 코드수정 전 백그라운드 kill. 자잘한 건 에이전트 위임.
- 잔차 품질 = 상태추정 품질. gym은 real vy/r 복원됨(커밋 6e83b12). 실차는 EKF 필요.
- `nominal_dynamics.py`가 잔차 계산 단일 소스. `scripts/extract_residuals.py`에 중복 미러 있으나 surgical 원칙으로 그대로 둠(Task 5에서 교차검증).
