# Phase B — MPCC EXTERNAL_COST 기반 재설계 (real-car best-in-class)

작성 2026-06-04. 목표 = 로컬 참조 MPCC 정독에서 나온 최고 기법(#1 선형진행 · #2 tire제약 · #5 convex-α LMPC · #3 GP · #4 식별타이어/μ)을 **co-tuned EXTERNAL_COST** 안에서 채택. 메모리 [[mpcc_reference_techniques]] [[alat_sweep_final]].

## 동기 (왜 B인가)
- 우리 컨트롤러는 뒤처진 게 아니라 advanced fork(GP-CasADi·multi-dt·sim-aligned·stuck복구). 버리지 않는다.
- 단 #1·#2·#5는 전부 EXTERNAL_COST(선형항·convex-α)를 요구. NLS 볼트온은 실패: 2026-06-04 slip-angle soft cap → 검증 baseline 19.84/0접촉이 19.88/**3접촉** 회귀, af_max 0.26→0.887(역효과). 튜닝된 NLS cost와 싸움.
- → real-car best-in-class엔 cost 축 교체가 불가피. "처음부터 재작성"은 아님(동역학·codegen·복구·제약 재사용).

## 비목표
- 동역학 모델 재작성 X (8-state Pacejka tanh 유지, B4서 식별타이어/μ만).
- SQP_RTI 포기 X (실시간 필수). convex-α는 slack+α warm-start로 RTI 양립(Racing-LMPC 분석).
- corridor/obstacle/stuck복구/codegen 위생 재작성 X (재사용).

## 단계 (각 단계 sweep 하네스 검증, 회귀 시 정지·롤백)

### B0 — 기반 전환 (de-risk 핵심, 새 기법 추가 전 필수 통과)
- `cost_type NONLINEAR_LS → CONVEX_OVER_NONLINEAR`(CONL: LS부 GN-Hessian 유지 + 선형항 허용). stage·terminal·kin·dyn 전부.
- ψ(r)=½ rᵀ W r 로 **현 cost 정확 재현** (r = 기존 y_expr).
- **검증기준: 현 baseline과 동일거동** (final dynamic 19.84s/0접촉 재현, solve_ms·feas 유사). 안 되면 여기서 멈춤.
- acados CONL API/Hessian 옵션 우리버전서 확인 필요.

### B1 — 선형 진행보상 (#1)
- ψ에 `−γ·p_v` 추가 (선형=convex). `q_p·(p_v−v_max)²` 제거 또는 약화. `p=v` 결합(#6, ṡ=p=v) → 퇴화해 제거.
- 기대: stall/centerline천장 돌파. γ는 sweep/BO로.

### B2 — co-tuned tire 제약 (#2)
- friction-ellipse `(E·Frx/Fz)²+(Fry/Fz)²≤(E·μ·Dr/Fz)²` + slip cap, **새 cost와 함께 재튜닝**(NLS 실패의 그 제약, 이번엔 균형 안에서). Liniger `constraints.cpp:57-157`.

### B3 — convex-α LMPC (#5)
- terminal = α 결정변수(α≥0,Σα=1, Σαⱼ·Qⱼ). convex-hull 등식 slack(`[20,20,2,...]`) + 지난α warm-start `zt=SS_next·α`. SS 48-96점/3랩·forward-time window·cost-to-go 로컬재영점. Racing-LMPC `racing_mpc.cpp:479-504`.

### B4 — sim2real (#3 #4)
- GP Jacobian 전치 fix(`transpose(1,0,2)`) + closed-form CasADi를 f_expl에 baking. 식별 비대칭타이어(ifac pacejka.yaml) + μ를 online param. RecordDataStrategy로 실차 데이터수집.

## 검증 / 위험관리
- 하네스: sweep 스크립트(launcher 패턴-self-kill 주의 [[alat_sweep_final]]), 고정윈도 다중랩, s-rollover lap, eval_run_quality. config마다 codegen 캐시 rm.
- 매 단계 NLS 코드 보존(가역). baseline 깨지면 롤백.
- 안전한 A-승리(a_lat=11.5 검증 −9.8%)는 직교 — 언제든 yaml 한 줄로 적용가능.

## 성공기준
- B0: 거동 재현(회귀 0).
- 전체: final 19.84s↓·0접촉 유지 + 실차 이식가능(grip honest). 각 단계가 baseline 대비 개선 or 중립.

---

## 진행 로그 (2026-06-04 세션)
- **B0 완료·커밋 `bc0825a`**: NLS→CONL. ψ=½rᵀWr·r=y_expr 정확재현 (final dynamic 22.12s/0접촉, NLS 22.00과 동일). solve +2.5ms(7ms, 25ms예산 내). GAUSS_NEWTON Hessian 유지.
- **B1 검증·revert**: 선형진행보상 CONL ψ=½rᵀWr−γ·r[5] (r[5]=q_p_scale·p_v, W[5]=0). final서 clean win 無 (grip-limited 맵). γ스윕 7.14/6|11.5/6: NLS 22.12·0|19.84·0, γ4 20.84·1|21.12·10, γ2.5 22.20·0|19.64·3, γ1 24.16·0|20.96·0. rand_a 교차검증=spawn 동결 inconclusive. → revert(깨끗한 B0 유지). B1은 **열린트랙용 옵션**(progress가 병목일 때). γ는 향후 BO 차원으로.

## B3 상세 구현계획 (다음 세션 — convex-α LMPC)
참조: Racing-LMPC-ROS2 `racing_mpc.cpp:479-504`, Berkeley `FTOCP.py:127-169`, `LMPC.py:134-181`. 우리 인프라: `mpc_core/lmpc/lap_database.py`·`safe_set.py`(존재, use_lmpc=false).

**B3.1 SS 인프라 업그레이드** (코드: lap_database/safe_set)
- K=10 → **num_ss_pts=48-96 (per_lap=16-32 × 3랩)**. 8-D convex hull이 x_N 담으려면 K=10 부족.
- cost-to-go = reverse step-count(이미 있음) + **로컬 재영점 `Q − Q[0]`** (query시, racing_mpc.cpp:280).
- **forward-time window 선택**: 현 Frenet s보다 ~N step 앞 P점, 매 cycle +1 전진 (Berkeley timeSS, LMPC.py:160). "ahead-of-me" 보장 = drift/wall 방지 핵심.

**B3.2 α 결정변수** (acados, setup_MPC)
- num_ss_pts개 α를 **extra control** (또는 terminal-stage 증강 state)로. box `α≥0`.
- 선형등식 **Σα=1**: 상수행 C로 (linear constraint).
- nx 커플링 **x_N − SS·α = 0**.

**B3.3 slack (SQP_RTI 양립 핵심)**
- 커플링을 hard 아닌 **L2 slack** (acados Zl/zl/idxs). per-state weight `[20,20,2,...]`(pos/yaw stiff, vel loose). 우리 hard-terminal 시도가 wedge난 그 fix.

**B3.4 terminal cost = Qᵀα** (선형). CONL ψ_e에 `+ Qᵀα` 또는 linear terminal. (B0의 CONL이 이걸 가능케 함.)

**B3.5 SS·Q를 terminal p_sym 확장** (대규모 슬롯). SS행렬(nx×num_ss_pts)+Q벡터+SS_next를 online param.

**B3.6 α warm-start** (mpc_node, 매 cycle): 지난 α 읽기 → `x_N_seed = SS_next·α` 로 terminal 초기화 + α도 이전값 init. RTI 1-iter가 hull 안에서 시작 → wedge 방지.

**B3.7 검증**: use_lmpc=true, final서 lap2+ 활성. LMPC가 더 빠른 코너line 찾나 (vs B0 22.12 / a_lat11.5 19.84). 회귀(drift/wall) 시 slack weight·SS크기·forward-window 튜닝.

**위험**: 큰 build(멀티턴). 각 B3.x가 독립 검증점. SS 인프라(B3.1)부터 단독 검증 후 acados 배선(B3.2-6).

## B3 Step 1 — state augmentation 정확한 edit-checklist (2026-06-04 진행)
**Step 1a 완료**: `_build_dynamic_model` return에 x_aug/f_aug/xdot_aug/alpha_aug/K_aug 추가 (available-but-unused, 빌드 무변). ★ 나머지는 **gate**(`use_dynamic AND self._lmpc_joint`)로 감싸 LMPC off면 nx=8 baseline 유지.

남은 edit points (acados_kinematic.py, K=10):
1. **setup_MPC dynamic 분기 (~575-590)**: `_lmpc_joint`면 `x=dyn['x_aug']; f_expl=dyn['f_aug']; xdot=dyn['xdot_aug']; nx=8+K; alpha=dyn['alpha_aug']` else 기존(nx=8). 비용 residual은 phys 심볼 그대로(x_aug 안 leaf라 무변).
2. **x0 (~1112)**: joint면 `ocp.constraints.x0` 대신 `idxbx_0=arange(8); lbx_0=ubx_0=zeros(8)` (α free at t=0).
3. **α bounds (~1136 idxbx)**: joint면 idxbx=`[3,4,5]+list(range(8,8+K))`, lbx/ubx에 `[0]*K`/`[1]*K` append.
4. **Σα=1**: con_h에 `ca.sum1(alpha)` 행 추가, lh=uh=1 (hard eq, 단순 simplex라 RTI OK). idxsh/slack은 이 행 제외(hard).
5. **solve loop set (~1764)**: joint면 `set(0,"lbx",x8); set(0,"ubx",x8)` (8 phys; idxbx_0=arange8이라 α free). 비-joint면 기존.
6. **X0 init (~1494,1514)**: joint면 X0 width 8+K, `X0[0,:8]=initial_state; X0[0,8:]=1.0/K` (uniform α seed). warm rollout도 8+K.
7. **traj fallback (~1817,1831)**: `np.tile(initial_state,(N+1,1))`는 8-wide → joint면 8+K-wide 패딩(α=1/K).
8. **traj 추출/mpc_node**: `traj[:, :8]`이 phys. `_lmpc_query_state=traj[-1,:8]` (LMPC_REBUILD 버그). GP B_d는 idx3,4,5 그대로 OK.

**Step 1 검증**: joint=on이지만 terminal은 아직 softmin(미사용 α) → α free·Σα=1뿐 → 거동 baseline 재현(LMPC_REBUILD "Step1 ✓ LMPC-off"). nx=18, solve +1ms.

## B3 Step 2 — convex-α terminal (CONL ψ_e)
- softmin terminal 제거. CONL ψ_e에 **linear cost-to-go `+ w_Q·(Qᵀα)`** (B0 CONL이 가능케 함; NLS는 sqrt였음) + **soft SS-anchor** residual `√(w_s)·W^½·(x_N[0:4] − SS·α)` (r_e에 4행 추가, ψ_e 제곱). SS=p[18:58] reshape(4,K), Q=p[58:68], α=x_aug[8:8+K].
- ★ soft anchor + corridor hard = 충돌이 drivable corridor로 해소(LMPC_REBUILD 핵심 안정성).
## B3 Step 3+ : SS infra(K↑·재영점·SS_next·forward window) → α warm-start(zt=SS_next·α) → 검증(use_lmpc on, lap2+).

## B3 Step 1 진행 (2026-06-04, 이어서)
**완료 (커밋됨, _lmpc_joint=False면 nx=8 baseline 무손상):**
- 1a: model에 x_aug/f_aug/xdot_aug/alpha_aug/K_aug.
- A: `self._lmpc_joint=False` flag (__init__).
- B: setup_MPC dynamic 분기 (joint면 x_aug/f_aug/xdot_aug, nx=8+K).
- C: x0 → partial `idxbx_0=arange(8)` (α free at t=0).
- D: α bounds idxbx [8..8+K-1] ∈ [0,1].
- E: con_h에 Σα 5번째 행 + lh/uh [1,1] (hard eq, idxsh는 [0,1,2,3] 유지).

**남은 (X0/warm-start, n_states-aware — 다음):**
- F(solve set): 불필요 확인됨 (idxbx_0=arange(8)라 set(0,"lbx",initial_state) 8-dim 그대로 OK).
- G: `self._nx_solver=nx` 저장(setup_MPC). X0 alloc(~1309,2111) `np.zeros((N+1, self._nx_solver))`.
- H: X0 init(~1543,1563) `X0[0,:self.n_states]=initial_state` + joint면 `X0[0,self.n_states:]=1/K`. 
- I: warm rollout(~1556,1570) `X0[k+1,:self.n_states]=xk[:n_states]+dt·deriv` + `X0[k+1,n_states:]=X0[k,n_states:]`(α const).
- J: traj fallback(~1866,1880) np.tile 폭 = _nx_solver. traj 추출은 `[:, :8]` phys. mpc_node `_lmpc_query_state=traj[-1,:8]`.
- **Step1 검증**: `_lmpc_joint=True` flip → 빌드+final 거동 baseline 재현(α free·Σα=1뿐, terminal 미사용) 확인. nx=18, solve +1ms.
- **Step2**: convex-α CONL terminal (linear Qᵀα + soft SS-anchor).
