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
