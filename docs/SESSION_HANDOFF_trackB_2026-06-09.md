# Track B 핸드오프 — kinematic+dynamic 둘 다 굴리기 / 실차 모델오차(GP vs B4') (2026-06-09)

> **다른 컴퓨터에서 이어서 작업하기 위한 자급식 문서.** 알고리즘 전체 설명은 `docs/MPCC_ALGORITHM_REFERENCE.md` 참고. 누적 컨텍스트는 메모리 `current_status_next.md`.

---

## 0. TL;DR — 지금 어디까지 왔나

- 작업을 **두 세션 트랙**으로 분리: **트랙 A = Mac/NUC 실차 포팅**(전용 세션, 아직 시작 안 함), **트랙 B = 이 문서**(kinematic+dynamic 둘 다 + 실차 모델오차 방식 결정).
- **이 세션(트랙 B)에서 한 것**: codegen 디렉토리 모델별 분리 ✅ (kinematic↔dynamic↔LMPC 전환 시 stale codegen 방지). TDD, GREEN.
- **다음(트랙 B)**: ① kinematic+LMPC guard, ② sim에서 kinematic·dynamic 둘 다 end-to-end 검증, ③ B-3 우선순위(full SQP, combined-slip, tire param 매칭…).
- ⚠️ **T7(실차 ws)는 이 세션에서 안 건드림.** 읽기 전용 확인만. 포팅은 트랙 A 전용.

---

## 1. 배경 — 차가 2대, compute도 2종

| 차 | 플랫폼 | 그립 | 비고 |
|---|---|---|---|
| 차1 | **Mac** (ARM, CPU/MPS, **CUDA 금지**) | (둘 중 하나) | acados Mac 재빌드 필요 |
| 차2 | **NUC** (x86 Linux, 표준) | (둘 중 하나) | acados 표준 빌드 |

**사용자 결정 (verbatim 취지):**
- "둘 다 혹시 모르니깐 **kinematic이랑 dynamic 다** 할거야" → 그립↔모델 고정 매핑 안 함. **두 차 모두 두 모델 다 빌드해서 둘 다 돌려보고 비교.**
- "일단 만들 때는 **NUC 기준**으로 만들어줘" → 개발/빌드 레퍼런스 플랫폼 = NUC(x86 Linux). (현재 dev/sim 박스가 x86 Linux라 일치.)
- 모델오차 학습(GP/B4')은 "**실차에서 돌리면 되니깐**" 하되, "**지금 차에서도 돌릴 수 있게 잘 만드는 게 중요**" → 실차 fleet(특히 저그립 차)의 1순위로 격상.

---

## 2. 트랙 B 핵심 결정 (근거 포함)

### 2-1. kinematic vs dynamic
- dynamic을 쓰는 *유일한 이유 = 타이어 슬립 포착*. **고그립=슬립小=kinematic이 오히려 정확**(추정 어려운 vy/r 상태 없음). 저그립=슬립大=dynamic 필요.
- 결정: **두 모델 다 빌드·비교** (위 사용자 결정). 그립별로 어느 게 나은지는 sim/실차에서 측정.

### 2-2. ★ GP·B4'는 둘 다 dynamic 전용
- GP residual·B4' error regression **둘 다 속도상태(vx/vy/r = idx 3,4,5)만 보정** (`acados_kinematic.py:1124-1130`, `gp_casadi_residual.py`). kinematic은 그 상태 자체가 없음 → **모델오차 보정은 dynamic(저그립) 차에만 해당.**

### 2-3. 실차 모델오차 — 층층이 (mu매칭 → B4' → GP)
1. **MPC 모델 tire/mu 파라미터 매칭** (저그립 차: 모델 mu→0.6). **학습 아님, 그냥 파라미터.** mismatch의 *체계적 대부분* 흡수. ★sim에서 지금 해야 함(학습 defer와 별개).
2. **B4'** = 남은 *체계적 상수 offset*. 순수 numpy → Mac/NUC 그냥 돎. sim 이득은 marginal(+3.6%)였지만 **실차 큰 mismatch에선 상수 offset이 쏠쏠**할 가능성 높음.
3. **GP** = 남은 *제어의존(δ,a_x)* 잔차. 천장 높음. **CasADi-export 경로(`use_gp_casadi`)**를 써야 — posterior mean을 acados solver에 C로 구워넣어 **Mac/NUC 동일 C 실행, torch 런타임 불필요.** 학습은 offline Linux+CUDA(두 차 공통), 차는 추론만, checkpoint device-무관.

### 2-4. ★ GP/B4'의 진짜 관문 = 깨끗한 vy/r 신호 (재발하던 천장)
- r(yaw rate) = 실차 IMU 자이로 **깨끗** ✓ / vx = odom **쓸만** ✓ / **vy = 어려움**(직접 센서 X → EKF/sideslip 추정). gym은 vy=0 하드코딩이라 sim 학습/검증도 막힘.
- 권장 GP 보정 대상: **vx+r 먼저(보수적), vy는 EKF 신뢰도 검증 후 추가.** (이 손잡이 미결정 — 사용자 차 상황에 달림.)

### 2-5. solver — acados 유지
- 우리 **compute-bound 아님**(solve 7~17ms ≪ 40ms/25Hz 예산), **solver 강건성도 이미 충분**(feas 100%, MINSTEP 0). crash-corner는 모델/기하 문제였지 solver 실패 아님.
- acados 유지 이유 = **40ms 안에 머물게 하면서 여유를 남겨 "추가"를 가능케 함**(IPOPT면 50~200ms로 즉시 초과). 여유 예산은 *solver*가 아니라 *모델 충실도*에 써야.

---

## 3. ★ B-3 보강 우선순위 (sim 단계, 학습#1 defer)

```
sim 단계:
1. full SQP (SQP_RTI iter 1→2~3)   — 코너 shake + 한계 선형화오차 직접 타격.
                                      ⚠️ solve×iter가 40ms(25Hz) 내인지 재확인.
2. combined-slip / friction-ellipse 제약  — 저그립 차 종+횡 타이어 한계.
                                      (지금 횡한계는 a_lat soft만 = 종+횡 결합 못 봄. 빈틈.)
3. 차별 tire/mu 파라미터 매칭  — 저그립=dynamic+mu0.6, 고그립=kinematic. (sim, 지금)
4. RVP 강화(q_v 속도비례↑) + raceline 의존 강화  — 짧은 호라이즌 유지하며 미리감속.
5. solve-time/제어율  — dynamic 부담 측정, 50Hz 가능? (full SQP와 예산 경합. Mac·NUC 각각 측정, 느린쪽=천장)
6. kinematic 경로 실동작 검증(고그립 차)  — 우선 LMPC off로.
── 실차 단계 (#1, defer) ──
7. GP residual(CasADi-export) + EKF(vy/r) + variance기반 constraint tightening.
8. autoreg 점진 탐사로 고속 데이터 안전수집.
```

> tanh vs full Pacejka 긴장 해소: **full Pacejka 대신 tanh 유지 + SQP iter 늘리기**(포화는 tanh, 선형화정확도는 iteration). full Pacejka는 RTI single-iter서 불안정(검증됨).

---

## 4. ✅ 이 세션에서 완료 — codegen 디렉토리 모델별 분리

**문제**: codegen 디렉토리가 고정(`/tmp/acados_codegen_evompcc`)이라, `use_dynamic`(nx 8↔5)·`use_lmpc`(nx 8↔18) 토글 시 **stale codegen = 조용히 틀린 모델** (반복해서 당한 함정). "둘 다 켜서 비교"하려면 잦은 전환이 필수라 이게 1순위 안전장치.

**수정** (`mpc_core/acados_kinematic.py`):
- 모듈함수 `codegen_paths(use_dynamic, lmpc_joint, nx_solver)` 추가 → `(export_dir, json)`를 모델/nx로 키잉:
  - kinematic → `/tmp/acados_codegen_evompcc_kin5`
  - dynamic(no lmpc) → `..._dyn8`
  - dynamic+lmpc → `..._dyn18_lmpc`
- codegen 사이트(구 1398행)에서 이 함수 호출.
- **효과**: 전환이 자동·안전(이제 수동 `rm -rf` 불필요), 같은 config는 warm-reuse.

**테스트**: `test/test_codegen_paths.py` (4개, GREEN). TDD RED→GREEN 확인.
```
cd src/nonlinear_mpc_acados && PYTHONPATH=. python3 -m pytest test/test_codegen_paths.py -q
```

⚠️ **미커밋.** 다른 컴퓨터로 가져가려면 **commit/push 필요** (브랜치 = `avoidance-restore`). 기존 stale `/tmp/acados_codegen_evompcc`는 한 번 `rm -rf` 해두면 깔끔.

---

## 5. ▶ 다음 즉시 할 일 (트랙 B 이어서)

### 5-1. kinematic+LMPC guard (다음 TDD 타겟)
**문제**: `mpc_node.py:1088` `_lmpc_joint = use_lmpc`. SS packing(`mpc_node.py:785-791`)은 `ss_states[3]=res.states[i,3]`을 **vx로 가정**. 하지만 kinematic 상태는 5-dim `[x,y,ψ,s,δ_prev]` → slot3 = **s(arc-length)**. → kinematic+LMPC면 **조용한 손상**(terminal cost에 s를 vx로 먹임). acados쪽은 `x_N_4`에서 kinematic이면 vx자리=0(line ~1002)이라 양쪽 불일치.
**수정안**: `use_dynamic=false`면 LMPC(joint/codegen/err_regr) **강제 off + 경고 로그** (가장 단순·정답). 위치: `mpc_node.py` ~1085-1101.
**TDD**: guard 로직을 순수함수로 빼서 "use_dynamic=False ⇒ lmpc_joint=False" 단언.

### 5-2. sim에서 두 모델 end-to-end 검증 (= "둘 다 시켜서 볼 거야")
```
# 항상 작업 전 백그라운드 kill (PID 기반). 단일 인스턴스 확인.
# dynamic (현 baseline):
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final use_dynamic:=true  use_lmpc:=true
# kinematic (검증 대상, LMPC off):
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final use_dynamic:=false use_lmpc:=false
```
- codegen 자동 분리되므로 전환 시 수동 삭제 불필요(§4). 첫 빌드만 ~30s.
- 확인: launch 성공, 랩 완주, STUCK/teleport, lap time, `[MPC-acados] ... KINEMATIC 5-state`/`DYNAMIC 8-state` 로그(model 토글 진짜 먹었는지 = `mpc_node.py:1113,1119-1128`).
- 비교는 **같은 LMPC 설정**(둘 다 off)으로 apples-to-apples 먼저.

### 5-3. 이후 B-3 §3 순서대로 (full SQP → combined-slip → tire/mu 매칭 …).

---

## 6. 핵심 파일 맵 / 게이트

| 무엇 | 위치 |
|---|---|
| 모델(dynamic+kinematic), cost, 제약, codegen | `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py` |
| ROS 노드(파라미터, SS packing, e_corr) | `.../mpc_node.py` |
| GP CasADi-export(임베디드 추론) | `.../mpc_core/gp_casadi_residual.py` |
| GP torch wrapper(비임베디드) | `.../mpc_core/gp_residual_wrapper.py` |
| B4' error regression | `.../mpc_core/lmpc/error_regression.py`, `lap_database.py`, `nominal_dynamics.py` |
| 메인 config | `src/nonlinear_mpc_acados/config/ddrx_unified_params.yaml` |
| sim 마찰 mu | `src/stack_master/config/SIM/sim_params.yaml` (현 0.6, 미커밋) |
| 알고리즘 전체 설명 | `docs/MPCC_ALGORITHM_REFERENCE.md` |

**현 config 토글** (`ddrx_unified_params.yaml`): `use_dynamic=true`, `dyn_tire_model=tanh`, `use_lmpc=true`, `use_gp_residual=false`, `use_gp_casadi=false`, `use_error_regression=false`, `max_speed=8`.

**게이트/함정**:
- 작업 전 백그라운드(sim/ros) 전부 kill, **PID 기반**(pgrep self-match 주의). 단일 인스턴스 확인.
- codegen은 이제 모델별 자동 분리(§4). 그래도 *cost weight/제약식* 같은 비-dim 변경은 같은 dir 재생성이라 의심되면 해당 `/tmp/acados_codegen_evompcc_*` 삭제.
- BO-best는 **적용 전 3런 검증**(저그립서 BO eval 노이즈로 재현 안 됨, a_lat 과하면 미끄러짐).
- 브랜치 = `avoidance-restore`. 커밋/푸시해야 다른 컴퓨터로 넘어감.

---

## 7. 트랙 A (포팅) — ✅ 포팅 키트 준비됨 (T7 분리돼도 OK)
- 타겟 = `<T7마운트>/ros2_ws/ros2_ws/src/IFAC2026_SH/src/control/nonlinear_mpcc/nonlinear_mpc_acados` (Mac/NUC 실차 ws, 05-27 구버전, 패키지명 동일).
- **★ `scripts/port_to_realcar.sh` 로 한 방에 옮김** (T7이 어디 마운트되든 타겟 경로만 인자로):
  ```bash
  # 미리보기(dry-run, 기본):
  scripts/port_to_realcar.sh <타겟_nonlinear_mpc_acados_경로>
  # 실제 적용:
  scripts/port_to_realcar.sh <타겟_nonlinear_mpc_acados_경로> --apply
  ```
  - **안전장치**: 기본 dry-run / 타겟 basename·package.xml 검증(엉뚱한 곳 거부) / rsync 범위가 그 패키지 dir로 고정 → **sibling 패키지(calibration/sensor/slam/system…) 절대 안 건드림**(검증 완료) / build·__pycache__·install 제외 / 알고리즘+핸드오프 문서 자동 투하.
- **적용 후 실차에서**: ① acados 재빌드(Mac=ARM·CUDA금지 / NUC=x86), `rm -rf /tmp/acados_codegen_evompcc*` 먼저 ② `colcon build --packages-select nonlinear_mpc_acados` ③ `config/ddrx_unified_params.yaml` 차별 검토(덮어써짐) ④ 차별 모델(저그립=dynamic+mu매칭 / 고그립=kinematic, LMPC 자동off).
- ⚠️ **이 세션에서 T7엔 안 씀**(읽기 전용 확인만). 실제 sync는 사용자가 트랙 A 세션에서 위 스크립트로.
