# mu 정직화 + Friction Ellipse + 제동 정직화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MPC 내부 모델이 실제 그립(μ)·제동한계를 정직하게 알게 한다 — dyn_mu 파라미터화 + friction-ellipse soft 제약 1행 + ref_v 제동/κ_eq 일관화.

**Architecture:** 그립 상수를 `model_policy.py`(의존성-프리) 단일 소스로 모으고, `acados_kinematic.py`는 소비 지점 4곳에서 clamp 헬퍼를 호출, h-제약에 ellipse 1행 추가(slack 확장), `track_loader.py`는 제동 a_long을 솔버 한계로 받는다. codegen 디렉토리는 mu까지 키잉해 stale codegen 함정 차단.

**Tech Stack:** Python, CasADi/acados, unittest. 스펙: `docs/superpowers/specs/2026-06-10-friction-ellipse-mu-design.md`

**검증 게이트(코드 태스크 후):** 유닛 → 회귀 sim(mu=1.0489, final, 3런) → 저그립 sim(mu=0.6, final2, 3런) → BO. 각 태스크마다 커밋, 게이트 통과 시 push + mpcc 레포 동기화.

**전제:** 모든 경로는 `/home/hmcl/IFAC2026_SH` 기준. 패키지 루트 = `src/nonlinear_mpc_acados`. 테스트 실행 패턴:
```bash
cd /home/hmcl/IFAC2026_SH && PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest nonlinear_mpc_acados.test.<모듈> -v
```
⚠️ 시작 전 백그라운드 sim/ROS 프로세스 전부 kill (PID 기반). ⚠️ 줄번호는 2b8f378 기준 — 밀릴 수 있으니 `old_string` 매칭으로 편집.

---

### Task 1: model_policy.py 그립 단일 소스 (TDD)

**Files:**
- Modify: `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/model_policy.py`
- Test: `src/nonlinear_mpc_acados/test/test_model_policy_grip.py` (생성)

- [ ] **Step 1: 실패하는 테스트 작성**

`src/nonlinear_mpc_acados/test/test_model_policy_grip.py` 생성:

```python
"""Grip single-source helpers — μ·g·η lateral limit + a_lat clamp + brake const.

2026-06-10 spec (friction-ellipse-mu-design): the controller must never plan on
more lateral grip than μ·g·η, and ref_v braking must use the SOLVER brake limit.

Run:
    PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest \
        nonlinear_mpc_acados.test.test_model_policy_grip -v
"""
from __future__ import annotations

import unittest

from nonlinear_mpc_acados.mpc_core.model_policy import (
    A_MIN_DYN, G_GRAV, clamp_a_lat_to_grip, grip_a_lat_limit)


class TestGripHelpers(unittest.TestCase):
    def test_brake_const_matches_solver(self):
        # acados_kinematic lbu[0] 와 단일 소스 — 솔버 제동한계 -3.0.
        self.assertEqual(A_MIN_DYN, -3.0)

    def test_grip_limit_value(self):
        # mu=0.6, η=0.95 → 0.6·9.81·0.95 = 5.5917
        self.assertAlmostEqual(grip_a_lat_limit(0.6, 0.95), 5.5917, places=4)
        self.assertAlmostEqual(G_GRAV, 9.81)

    def test_clamp_above_limit(self):
        # 요청 7.1445 > 한계 5.5917 → clamp + flag
        eff, clamped = clamp_a_lat_to_grip(7.1445, mu=0.6, ellipse_frac=0.95)
        self.assertAlmostEqual(eff, 5.5917, places=4)
        self.assertTrue(clamped)

    def test_clamp_below_limit_passthrough(self):
        # 고그립(mu=1.0489): 한계 9.777 > 요청 7.1445 → 그대로 (기존 동작 보존)
        eff, clamped = clamp_a_lat_to_grip(7.1445, mu=1.0489, ellipse_frac=0.95)
        self.assertAlmostEqual(eff, 7.1445, places=4)
        self.assertFalse(clamped)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run: `cd /home/hmcl/IFAC2026_SH && PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest nonlinear_mpc_acados.test.test_model_policy_grip -v`
Expected: FAIL — `ImportError: cannot import name 'A_MIN_DYN'`

- [ ] **Step 3: 구현**

`model_policy.py` 파일 끝(`effective_lmpc` 함수 뒤)에 추가:

```python
# ─── Grip single source (2026-06-10 friction-ellipse-mu spec) ───────────────
G_GRAV = 9.81
# Solver longitudinal brake limit [m/s²]. MUST stay in sync with
# acados_kinematic lbu[0] (which imports this const — single source).
A_MIN_DYN = -3.0


def grip_a_lat_limit(mu, ellipse_frac=0.95):
    """Physical lateral-accel ceiling a_lim = μ·g·η (η = ellipse headroom)."""
    return float(mu) * G_GRAV * float(ellipse_frac)


def clamp_a_lat_to_grip(a_lat_safe, mu, ellipse_frac=0.95):
    """Clamp a requested a_lat_safe to the physical μ·g·η ceiling.

    Returns (effective_a_lat, clamped). BO/yaml can request any a_lat — the
    speed profile must never be built on grip the tire cannot deliver
    (mu=0.6 BO-best non-reproduction root cause, 2026-06-09/10).
    """
    lim = grip_a_lat_limit(mu, ellipse_frac)
    a = float(a_lat_safe)
    return (min(a, lim), a > lim)
```

- [ ] **Step 4: 통과 확인**

Run: 위와 동일. Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/hmcl/IFAC2026_SH && git add src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/model_policy.py src/nonlinear_mpc_acados/test/test_model_policy_grip.py && git commit -m "feat(grip): model_policy 그립 단일소스 — μgη 한계 + a_lat clamp + A_MIN_DYN"
```

---

### Task 2: codegen_paths에 dyn_mu 키잉 (TDD)

dyn_mu는 f_expl(tanh 타이어)·ellipse에 **codegen 시점에 박히는 상수** — mu가 다른데 같은 디렉토리를 재사용하면 stale codegen으로 잘못된 모델이 돈다 (반복돼온 함정).

**Files:**
- Modify: `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py` (`codegen_paths`, 라인 ~30)
- Test: `src/nonlinear_mpc_acados/test/test_codegen_paths.py` (테스트 추가)

- [ ] **Step 1: 실패하는 테스트 추가**

`test_codegen_paths.py`의 `TestCodegenPaths` 클래스에 추가:

```python
    def test_mu_keys_the_codegen(self):
        # dyn_mu 는 tanh 타이어/ellipse 에 codegen-time 으로 박힌다 —
        # mu 0.6 vs 1.0489 가 같은 dir 을 재사용하면 stale 모델.
        lo = codegen_paths(use_dynamic=True, lmpc_joint=False, nx_solver=8,
                           dyn_mu=0.6)
        hi = codegen_paths(use_dynamic=True, lmpc_joint=False, nx_solver=8,
                           dyn_mu=1.0489)
        self.assertNotEqual(lo[0], hi[0], "mu must key the export dir")
        self.assertNotEqual(lo[1], hi[1], "mu must key the json")

    def test_mu_default_keeps_legacy_tag(self):
        # mu 미지정(레거시 호출) → 기존 태그 그대로 (경로 호환).
        legacy = codegen_paths(use_dynamic=True, lmpc_joint=False, nx_solver=8)
        self.assertEqual(legacy[0], "/tmp/acados_codegen_evompcc_dyn8")
```

- [ ] **Step 2: 실패 확인**

Run: `PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest nonlinear_mpc_acados.test.test_codegen_paths -v`
Expected: FAIL — `TypeError: codegen_paths() got an unexpected keyword argument 'dyn_mu'`

- [ ] **Step 3: 구현**

`acados_kinematic.py`의 `codegen_paths`를 수정 — 시그니처와 tag 줄:

```python
def codegen_paths(use_dynamic, lmpc_joint, nx_solver, dyn_mu=None):
```

기존 `tag = ...` 줄을:

```python
    tag = f"{'dyn' if use_dynamic else 'kin'}{int(nx_solver)}{'_lmpc' if lmpc_joint else ''}"
    if dyn_mu is not None:
        # μ is baked into the tanh tire + friction-ellipse codegen — key it
        # so switching dyn_mu can never reuse a stale build (mu0p600 etc.).
        tag += f"_mu{float(dyn_mu):.3f}".replace('.', 'p')
    return (f"/tmp/acados_codegen_evompcc_{tag}",
            f"/tmp/acados_ocp_evompcc_{tag}.json")
```

docstring의 NOTE 단락에서 "baked cost weights — are NOT in the tag" 목록은 그대로 두되 한 줄 추가: `dyn_mu IS keyed (2026-06-10).`

그리고 `setup_MPC` 안의 `codegen_paths(...)` **호출부**를 찾아(`grep -n "codegen_paths(" acados_kinematic.py`) `dyn_mu=float(self.dyn_mu)` 인자를 추가:

```python
        export_dir, json_path = codegen_paths(self.use_dynamic, self._lmpc_joint,
                                              nx_solver, dyn_mu=float(self.dyn_mu))
```
(호출부 변수명이 다르면 해당 이름 유지 — 인자만 추가.)

- [ ] **Step 4: 통과 확인**

Run: 위와 동일. Expected: 기존 4 + 신규 2 = 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py src/nonlinear_mpc_acados/test/test_codegen_paths.py && git commit -m "feat(grip): codegen 디렉토리 dyn_mu 키잉 — mu 전환 stale codegen 차단"
```

---

### Task 3: dyn_mu/ellipse_frac 파라미터 배선 (node→mpc) + a_min_dyn 단일소스

**Files:**
- Modify: `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_node.py` (declare ~236, attr push ~1060)
- Modify: `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py` (ellipse_frac default ~106, a_min_dyn ~1249)
- Modify: `src/nonlinear_mpc_acados/config/ddrx_unified_params.yaml` (~86)
- Modify: `src/stack_master/launch/full_sim.launch.py` (override 루프 ~209, DeclareLaunchArgument ~341)

- [ ] **Step 1: acados_kinematic 기본값 + import**

`acados_kinematic.py` 상단 import(`from ._ros_compat import ...` 근처)에 추가:

```python
from .model_policy import A_MIN_DYN, clamp_a_lat_to_grip
```

라인 ~106 `self.dyn_mu   = 1.0489` 바로 아래에 추가:

```python
        self.ellipse_frac = 0.95  # friction ellipse headroom η (a_lim = μ·g·η)
```

라인 ~1249 `a_min_dyn = -3.0` 을:

```python
        a_min_dyn = A_MIN_DYN   # 솔버 제동한계 — model_policy 단일소스 (ref_v a_long 과 동기)
```

- [ ] **Step 2: mpc_node declare**

`mpc_node.py` 라인 ~236 `self.declare_parameter('dyn_tire_model', 'linear')` 바로 아래에 추가:

```python
        # ── 그립 정직화 (2026-06-10) ──
        # dyn_mu: 모델 타이어 μ. sim_params.yaml 의 mu(또는 실차 노면)와 일치시켜야
        # 모델이 정직 — 저그립 실험은 dyn_mu:=0.6. codegen-time 상수 (mu 가 codegen
        # 디렉토리 태그에 키잉되므로 전환 시 수동 rm 불필요).
        self.declare_parameter('dyn_mu', 1.0489)
        # ellipse_frac: friction ellipse headroom η. a_lim = μ·g·η.
        self.declare_parameter('ellipse_frac', 0.95)
```

- [ ] **Step 3: attr push — 반드시 set_track_data보다 먼저**

`set_track_data()`가 brake-aware κ_eq에서 `dyn_mu`를 소비(Task 5)하므로, push는 라인 ~1060 `self.mpc.set_initial_params(param, vheid, is_ot)` **바로 다음**(= `set_track_data(` 호출 **앞**)에 삽입:

```python
        # 그립 정직화 — set_track_data(κ_eq/bf)·setup_MPC(ellipse/codegen tag) 둘 다
        # dyn_mu 를 소비하므로 track 빌드 전에 push (push 순서 의존).
        self.mpc.dyn_mu = float(self.get_parameter('dyn_mu').value)
        self.mpc.ellipse_frac = float(self.get_parameter('ellipse_frac').value)
```

(기존 ~1071 `self.mpc.use_dynamic = ...` 블록은 그대로 둔다.)

- [ ] **Step 4: yaml 노출**

`config/ddrx_unified_params.yaml` 라인 ~87 `dyn_tire_model: tanh` 아래에 추가:

```yaml
    # 모델 타이어 μ — sim_params.yaml mu(또는 실차 노면)와 일치시킬 것 (2026-06-10 그립 정직화)
    dyn_mu: 1.0489
    # friction ellipse headroom η (a_lim = μ·g·η). 1.0=물리한계 꽉 채움.
    ellipse_frac: 0.95
```

- [ ] **Step 5: launch 인자 노출**

`src/stack_master/launch/full_sim.launch.py` 라인 ~209 `for _k in ("use_dynamic", "use_lmpc"):` 를:

```python
        for _k in ("use_dynamic", "use_lmpc", "dyn_mu"):
```

라인 ~341 근처 `DeclareLaunchArgument("use_dynamic", ...)` 옆에 추가:

```python
        DeclareLaunchArgument("dyn_mu", default_value="",
                              description="모델 타이어 μ override (빈값=yaml). sim mu와 일치시킬 것"),
```

⚠️ override 루프가 빈 문자열→yaml 폴백을 어떻게 처리하는지 기존 use_dynamic 코드(~239)를 그대로 따른다. float 캐스팅이 필요한 구조면 동일 패턴으로 처리(use_dynamic이 bool 캐스팅하는 자리). 편집 후 해당 블록을 읽고 dyn_mu가 빈값일 때 yaml값이 살아남는지 눈으로 확인.

- [ ] **Step 6: 스모크 확인**

```bash
cd /home/hmcl/IFAC2026_SH && python3 -c "
import ast,sys
src=open('src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_node.py').read()
assert \"declare_parameter('dyn_mu', 1.0489)\" in src
assert src.index('self.mpc.dyn_mu') < src.index('self.mpc.set_track_data'), 'push must precede set_track_data'
print('wiring OK')"
```
Expected: `wiring OK`

- [ ] **Step 7: Commit**

```bash
git add -A src/nonlinear_mpc_acados src/stack_master/launch/full_sim.launch.py && git commit -m "feat(grip): dyn_mu/ellipse_frac param 배선 (yaml+launch) + a_min_dyn 단일소스"
```

---

### Task 4: a_lat_safe 소비 지점 4곳 μ-clamp

**Files:**
- Modify: `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py` (4곳)

- [ ] **Step 1: clamp 접근자 메서드 추가**

`MPC` 클래스에 (라인 ~210 `self.a_lat_safe_live = 6.0` 정의가 속한 클래스; `codegen_paths` 함수 말고 클래스 내부, 예: `set_track_data` 정의 직전) 추가:

```python
    def a_lat_safe_eff(self):
        """a_lat_safe_live clamped to physical grip μ·g·η.

        BO/yaml/rqt 가 어떤 값을 요청하든 속도 프로필·vcap·a_lat 제약이
        타이어가 못 내는 그립 위에 세워지지 않게 하는 단일 게이트.
        Live-safe: 호출 시점의 a_lat_safe_live/dyn_mu 로 매번 계산.
        """
        eff, clamped = clamp_a_lat_to_grip(
            self.a_lat_safe_live, self.dyn_mu, self.ellipse_frac)
        if clamped and not getattr(self, '_alat_clamp_warned', False):
            self._alat_clamp_warned = True
            self._log.warning(
                f"[MPC-acados] a_lat_safe_live {float(self.a_lat_safe_live):.2f} > "
                f"μgη {eff:.2f} (mu={float(self.dyn_mu):.3f}) → clamped. "
                "BO 탐색 상한이 물리 그립에 묶임 (의도된 동작).")
        return eff
```

- [ ] **Step 2: 소비 지점 교체 (4곳)**

①  라인 ~386 (`set_track_data`의 brake-aware κ_eq):

```python
        _bf = 0.7   # brake safety factor: a_brake = bf·a_lat (더 일찍 제동 → late-brake STUCK↓)
```
→
```python
        # bf = a_brake/a_lat (κ_eq 유도식 그대로). 고정 0.7은 a_lat_safe=5 에서
        # 제동 3.5 m/s² 가정 = 솔버 한계(3.0) 초과 낙관 — 솔버 값으로 일관화.
        _bf = min(1.0, abs(A_MIN_DYN) / max(1e-3, self.a_lat_safe_eff()))
```

② 라인 ~1292 (a_lat hard cap backstop):

```python
        a_lat_max = max(8.0, float(self.a_lat_safe_live) + 1.0)
```
→
```python
        a_lat_max = max(8.0, float(self.a_lat_safe_eff()) + 1.0)
```
(다음 줄 로그 f-string의 `a_lat_safe=` 값도 `float(self.a_lat_safe_eff())`로 교체.)

③ 라인 ~1810 (per-stage vcap):

```python
                _vcap = math.sqrt(float(self.a_lat_safe_live) / (_absk + 1e-3))
```
→
```python
                _vcap = math.sqrt(float(self.a_lat_safe_eff()) / (_absk + 1e-3))
```

④ 라인 ~1869 및 ~1332 (runtime p_arr / 초기 parameter_values 의 A_LAT_SAFE 슬롯):

```python
            p_arr[10] = float(self.a_lat_safe_live)
```
→
```python
            p_arr[10] = float(self.a_lat_safe_eff())
```

```python
        ocp.parameter_values[10] = self.a_lat_safe_live     # A_LAT_SAFE
```
→
```python
        ocp.parameter_values[10] = self.a_lat_safe_eff()    # A_LAT_SAFE (μ-clamped)
```

- [ ] **Step 3: 잔여 소비처 grep 확인**

```bash
grep -n "a_lat_safe_live" src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py
```
Expected: 남는 곳 = ① 기본값 정의(~202) ② `a_lat_safe_eff` 내부 ③ 로그 문자열 정도. **연산에 raw 값을 쓰는 줄이 남아 있으면 안 됨** (있으면 같은 방식으로 교체).

- [ ] **Step 4: 유닛 전체 회귀**

```bash
PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest discover -s src/nonlinear_mpc_acados -p "test_*.py" -v 2>&1 | tail -5
```
Expected: 전부 PASS (acados import 불가 테스트는 기존과 동일하게 skip 처리되어 있음).

- [ ] **Step 5: Commit**

```bash
git add src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py && git commit -m "feat(grip): a_lat_safe 소비 4곳 μgη clamp (vcap/backstop/p_arr/bf) — bf 솔버제동 일관화"
```

---

### Task 5: friction ellipse h-제약 1행 + slack 확장

**Files:**
- Modify: `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py` (con_h_expr ~1106, lh/uh/idxsh/zl ~1295)

- [ ] **Step 1: con_h_expr에 ellipse 행 삽입**

라인 ~1106의:

```python
        a_lat = a_lat_expr
        if self._lmpc_joint and alpha_sym is not None:
            # B3: α simplex Σα=1 as a 5th h-row (hard eq, NOT slacked).
            model_ac.con_h_expr = ca.vertcat(h_obs, h_corridor_top, h_corridor_bot, a_lat, ca.sum1(alpha_sym))
        else:
            model_ac.con_h_expr = ca.vertcat(h_obs, h_corridor_top, h_corridor_bot, a_lat)
```
→
```python
        a_lat = a_lat_expr
        # ---- (4) friction ellipse (2026-06-10 spec) ----
        # (a_x/a_lim)² + (a_lat/a_lim)² ≤ 1, a_lim = μ·g·η. u[0]=a_x 는 unified
        # 입력 레이아웃에서 양 모드 공통. 제동·가속 중 가용 a_lat 이 자동 감소
        # (combined-slip). 고그립(μ=1.0489)에선 a_lim≈9.8 → 거의 안 물림 = 기존
        # 동작 보존; 저그립(μ=0.6)에선 a_lim≈5.6 = 실질 한계.
        _a_lim = max(1e-3, float(self.dyn_mu) * 9.81 * float(self.ellipse_frac))
        h_ellipse = (u[0] / _a_lim) ** 2 + (a_lat / _a_lim) ** 2
        if self._lmpc_joint and alpha_sym is not None:
            # B3: α simplex Σα=1 — 마지막 h-row (hard eq, NOT slacked).
            model_ac.con_h_expr = ca.vertcat(h_obs, h_corridor_top, h_corridor_bot, a_lat, h_ellipse, ca.sum1(alpha_sym))
        else:
            model_ac.con_h_expr = ca.vertcat(h_obs, h_corridor_top, h_corridor_bot, a_lat, h_ellipse)
```

⚠️ `u`가 이 스코프에서 입력 심볼인지 확인(`model_ac.u = u` 가 같은 함수 아래쪽에 있음). 이름이 다르면 그 이름 사용.

- [ ] **Step 2: lh/uh/idxsh/slack 확장**

라인 ~1295의 h-bounds 블록을:

```python
        # h order: [h_obs, h_corridor_top, h_corridor_bot, a_lat, (Σα if joint)]
        if self._lmpc_joint:
            ocp.constraints.lh = np.array([0.0, 0.0, 0.0, -a_lat_max, 1.0])  # Σα=1 eq
            ocp.constraints.uh = np.array([1e15, 1e15, 1e15, a_lat_max, 1.0])
        else:
            ocp.constraints.lh = np.array([0.0, 0.0, 0.0, -a_lat_max])
            ocp.constraints.uh = np.array([1e15, 1e15, 1e15, a_lat_max])
```
→
```python
        # h order: [h_obs, corr_top, corr_bot, a_lat, ellipse, (Σα if joint)]
        # ellipse 행: 0 ≤ h ≤ 1 (h 는 제곱합이라 하한 0 은 자연 충족 — 무해).
        if self._lmpc_joint:
            ocp.constraints.lh = np.array([0.0, 0.0, 0.0, -a_lat_max, 0.0, 1.0])  # Σα=1 eq
            ocp.constraints.uh = np.array([1e15, 1e15, 1e15, a_lat_max, 1.0, 1.0])
        else:
            ocp.constraints.lh = np.array([0.0, 0.0, 0.0, -a_lat_max, 0.0])
            ocp.constraints.uh = np.array([1e15, 1e15, 1e15, a_lat_max, 1.0])
```

이어지는 slack 블록을:

```python
        ocp.constraints.idxsh = np.array([0, 1, 2, 3])
        ns = 4
```
→
```python
        ocp.constraints.idxsh = np.array([0, 1, 2, 3, 4])   # ellipse(4) 포함, Σα 비slack
        ns = 5
```

zl/zu/Zl/Zu 배열을 (스펙: ellipse slack 가중치 = a_lat 행과 동일 zl=50/Zl=15):

```python
        ocp.cost.zl = np.array([40.0, 20.0, 20.0, 50.0])
        ocp.cost.zu = np.array([40.0, 20.0, 20.0, 50.0])
        ocp.cost.Zl = np.array([30.0, 15.0, 15.0, 15.0])
        ocp.cost.Zu = np.array([30.0, 15.0, 15.0, 15.0])
```
→
```python
        #   idx 4 (ellipse):     zl=50,  Zl=15   (a_lat 행과 동일 — spec §2)
        ocp.cost.zl = np.array([40.0, 20.0, 20.0, 50.0, 50.0])
        ocp.cost.zu = np.array([40.0, 20.0, 20.0, 50.0, 50.0])
        ocp.cost.Zl = np.array([30.0, 15.0, 15.0, 15.0, 15.0])
        ocp.cost.Zu = np.array([30.0, 15.0, 15.0, 15.0, 15.0])
```

- [ ] **Step 3: h-row 인덱스 잔여 참조 확인**

Σα 행 인덱스가 4→5로 밀렸다. 다른 코드가 h-row 인덱스를 참조하는지 확인:

```bash
grep -n "con_h\|lsh\|ush\|idxsh\|nsh" src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py | grep -v "^\s*#"
```
`lsh`/`ush`는 `np.zeros(ns)` 패턴이면 자동 확장 — ns=5 반영 확인. 그 외 h-row 번호 하드코딩 발견 시(예: terminal con_h, solver.set 의 h 인덱스) 같은 순서로 +1 보정.

- [ ] **Step 4: 유닛 전체 + 스모크**

```bash
PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest discover -s src/nonlinear_mpc_acados -p "test_*.py" 2>&1 | tail -3
```
Expected: 전부 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_core/acados_kinematic.py && git commit -m "feat(grip): friction ellipse soft h-제약 — (a_x/μgη)²+(a_lat/μgη)²≤1, slack 5행 확장"
```

---

### Task 6: ref_v 제동 정직화 — track_loader a_long (TDD)

**Files:**
- Modify: `src/nonlinear_mpc_acados/nonlinear_mpc_acados/track_loader.py` (시그니처 ~179, a_long ~278)
- Modify: `src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_node.py` (build_track 호출 ~1022)
- Test: `src/nonlinear_mpc_acados/test/test_brake_profile.py` (생성)

- [ ] **Step 1: 실패하는 테스트 작성**

`src/nonlinear_mpc_acados/test/test_brake_profile.py` 생성:

```python
"""ref_v forward-backward brake profile honesty — a_long must be the SOLVER
brake limit (|A_MIN_DYN|=3.0), not the lateral a_lat_max proxy (7.14 = 2.4×
optimistic → corner entry overspeed → the final2 3-corner stuck family).

Run:
    PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest \
        nonlinear_mpc_acados.test.test_brake_profile -v
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from nonlinear_mpc_acados.track_loader import build_track_from_wpnts


def _mk_wpnts(n=60, r_straight=1e9):
    """직선 50pt + 급코너(κ=2.0) 10pt 루프 흉내 — 코너 전 제동 프로파일 검증용.
    Wpnt 메시지 대신 SimpleNamespace (필드 호환)."""
    wp = []
    for i in range(n):
        ang = 2.0 * math.pi * i / n
        x, y = 10.0 * math.cos(ang), 10.0 * math.sin(ang)
        kappa = 2.0 if (n - 10) <= i else 0.01   # 마지막 10pt = 급코너
        wp.append(SimpleNamespace(
            x_m=x, y_m=y,
            psi_rad=ang + math.pi / 2.0, psi_centerline_rad=0.0,
            d_left=0.8, d_right=0.8, vx_mps=8.0, kappa_radpm=kappa))
    return wp


class TestBrakeProfileHonesty(unittest.TestCase):
    def _refv(self, a_long_max):
        td = build_track_from_wpnts(
            _mk_wpnts(), default_v=8.0, a_lat_max=7.1445,
            corridor_half_width=0.0,        # use_fixed_corridor=False → 프로파일 적용
            a_long_max=a_long_max)
        return np.asarray(td.raw_ref_v if hasattr(td, 'raw_ref_v') else td.ref_v,
                          dtype=float)

    def test_honest_brakes_earlier_than_optimistic(self):
        v3 = self._refv(3.0)
        v7 = self._refv(7.1445)
        # 정직한 제동(3.0)은 낙관(7.14)보다 어디서도 빠르면 안 되고,
        self.assertTrue(np.all(v3 <= v7 + 1e-9))
        # 코너 앞 어딘가에서는 분명히 더 일찍(더 낮게) 감속해야 한다.
        self.assertTrue(np.any(v3 < v7 - 0.2),
                        f"honest profile identical to optimistic: {v3} vs {v7}")

    def test_brake_rate_within_solver_limit(self):
        v = self._refv(3.0)
        # 인접 포인트 간 감속이 v²=v'²+2·a·ds (a=3.0+수치여유) 이내인지
        wp = _mk_wpnts()
        n = len(wp)
        for i in range(n):
            j = (i + 1) % n
            ds = math.hypot(wp[j].x_m - wp[i].x_m, wp[j].y_m - wp[i].y_m)
            if v[i] > v[j]:   # braking segment
                a_req = (v[i] ** 2 - v[j] ** 2) / (2.0 * max(ds, 1e-6))
                self.assertLessEqual(a_req, 3.0 + 0.15,
                                     f"i={i}: implied brake {a_req:.2f} > 3.0")


if __name__ == '__main__':
    unittest.main()
```

⚠️ `TrackData`의 ref_v 접근 필드명은 구현 확인 후 보정 (`td.ref_v`/`lut_ref_v`/`raw_ref_v` — `track_loader.py`의 `_build_track` 반환 구조를 읽고 프로파일이 들어가는 배열 속성으로 assert).

- [ ] **Step 2: 실패 확인**

Run: `PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest nonlinear_mpc_acados.test.test_brake_profile -v`
Expected: FAIL — `TypeError: build_track_from_wpnts() got an unexpected keyword argument 'a_long_max'`

- [ ] **Step 3: track_loader 구현**

시그니처(~179)에 추가:

```python
                           a_lat_max: float = 6.0,
                           a_long_max: float = 3.0,
```

라인 ~278:

```python
        a_long = float(a_lat_max)   # longitudinal accel/brake limit (g-g proxy)
```
→
```python
        # 제동 정직화 (2026-06-10): a_lat_max(=7.14) g-g proxy 는 솔버 제동한계
        # (-3.0)보다 2.4× 낙관 → 코너 진입 과속의 직접 경로였음. 솔버와 동일한
        # 종방향 한계를 명시적으로 받는다 (mpc_node 가 |A_MIN_DYN| 전달).
        a_long = float(a_long_max)
```

- [ ] **Step 4: mpc_node 호출부**

`mpc_node.py` 상단 import 줄(`from .mpc_core.model_policy import effective_lmpc`가 함수 내 local import이므로, 파일 상단 `from .track_loader import ...` 아래)에 추가:

```python
from .mpc_core.model_policy import A_MIN_DYN, clamp_a_lat_to_grip
```

라인 ~1022 `build_track_from_wpnts(` 호출에서 `a_lat_max=` 인자를:

```python
                a_lat_max=float(self.get_parameter('a_lat_safe_live').value),
```
→
```python
                a_lat_max=self._a_lat_eff_for_track(),
                a_long_max=abs(A_MIN_DYN),
```

그리고 같은 클래스에 헬퍼 추가 (예: `_build_param_dict` 정의 앞):

```python
    def _a_lat_eff_for_track(self) -> float:
        """track ref_v 용 a_lat — μgη clamp (BO 가 뭘 요청하든 물리 그립 상한)."""
        eff, clamped = clamp_a_lat_to_grip(
            float(self.get_parameter('a_lat_safe_live').value),
            float(self.get_parameter('dyn_mu').value),
            float(self.get_parameter('ellipse_frac').value))
        if clamped:
            self.get_logger().warn(
                f"[grip] a_lat_safe_live > μgη → track ref_v 는 {eff:.2f} 로 clamp")
        return eff
```

⚠️ `build_track_from_wpnts` 호출이 mpc_node에 2곳 이상인지 확인(`grep -n "build_track_from_wpnts" mpc_node.py`) — CSV fallback 경로(load_track)는 별도라 무관, wpnt 경로가 복수면 전부 동일 적용.

- [ ] **Step 5: 통과 확인 + 전체 유닛**

```bash
PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest nonlinear_mpc_acados.test.test_brake_profile -v
PYTHONPATH=src/nonlinear_mpc_acados python3 -m unittest discover -s src/nonlinear_mpc_acados -p "test_*.py" 2>&1 | tail -3
```
Expected: 신규 2 PASS + 전체 PASS (기존 track 테스트가 a_long 변화로 깨지면 해당 테스트의 기대값을 정직 프로파일로 갱신 — 단 "느려졌다"는 이유의 완화는 금지).

- [ ] **Step 6: Commit**

```bash
git add src/nonlinear_mpc_acados/nonlinear_mpc_acados/track_loader.py src/nonlinear_mpc_acados/nonlinear_mpc_acados/mpc_node.py src/nonlinear_mpc_acados/test/test_brake_profile.py && git commit -m "feat(grip): ref_v 제동 정직화 — a_long=|A_MIN_DYN| + track a_lat μ-clamp"
```

---

### Task 7: 회귀 게이트 — mu=1.0489, final, 3런 (기존 동작 보존 확인)

**Files:** 코드 변경 없음 (검증 전용)

- [ ] **Step 1: 환경 준비**

```bash
# 백그라운드 sim/ROS 전부 kill (PID 기반 — pkill 패턴 self-kill 함정 주의)
ps aux | grep -E "full_sim|mpc_node|gym_bridge|rviz" | grep -v grep
# sim_params.yaml mu 를 1.0489 로 임시 변경 (실험 후 0.6 복원!)
# config/ddrx_unified_params.yaml: dyn_mu 1.0489 확인 (디폴트)
cd /home/hmcl/IFAC2026_SH && colcon build --packages-select nonlinear_mpc_acados stack_master --symlink-install 2>&1 | tail -2
```

- [ ] **Step 2: 3런 (각 ~9분)**

```bash
source install/setup.bash && ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final
```
런 간 PID kill → 재실행. 각 런 로그에서:
- `[mpc] ACTUAL MODEL FOR CODEGEN = DYNAMIC 8-state, tire=tanh`
- codegen 디렉토리가 `_mu1p049` 태그인지: `ls /tmp/acados_codegen_evompcc*`
- `grep -c stuck-recover` = 0, lap ~21s (기존 21.0–21.9s 노이즈 범위)

- [ ] **Step 3: 판정**

PASS 기준: 3런 모두 STUCK 0 · lap 21.0–22.5s · feas 정상. ellipse는 μ=1.0489에서 거의 안 물려야 함(기존 동작 보존이 설계 의도). bf 변화(0.7→0.42)로 소폭 느려질 수 있음 — 22.5s 초과 시 사용자에게 보고 후 진행 여부 결정.

- [ ] **Step 4: Commit (게이트 기록) + push**

```bash
git commit --allow-empty -m "gate: 회귀 3런 PASS (mu=1.0489 final, STUCK0, lap <기록>)" && git push origin avoidance-restore
```

---

### Task 8: 저그립 게이트 — mu=0.6 + dyn_mu=0.6, final2, 3런 (본 게임)

- [ ] **Step 1: 설정**

```bash
# sim_params.yaml: mu 0.6 복원 (Task 7 에서 올렸던 것)
# 런치 인자로 모델도 정직하게:
ros2 launch stack_master full_sim.launch.py mode:=mpcc map:=final2 dyn_mu:=0.6
```
시작 로그에서 `μgη` clamp 경고 확인 (a_lat_safe 7.14 > 5.59 → clamp 발동이 정상).

- [ ] **Step 2: 3런 분석**

```bash
cd ~/analysis_tmp && python3 stuck_cluster.py ~/mpc_logs/mpc_<ts>.csv
```
PASS 기준: **3런 STUCK 0 · 접촉 0** (기준선: 같은 조건에서 3코너 stuck n=25/13/29였음). 랩타임은 느려도 됨 — 이 게이트는 안정성.

- [ ] **Step 3: 부분 실패 시**

stuck이 줄었지만 0이 아니면: 클러스터 위치가 기존 3코너인지 새 위치인지 구분해 보고. ellipse_frac 0.95→0.90, 또는 corridor_v_floor 등 기존 안전 knob 조합은 **사용자 보고 후** 결정 (즉흥 knob 금지).

- [ ] **Step 4: Commit + push**

```bash
git commit --allow-empty -m "gate: 저그립 3런 PASS (mu=0.6 final2, STUCK <기록>)" && git push origin avoidance-restore
```

---

### Task 9: mpcc 배포 레포 동기화 + GitHub push

- [ ] **Step 1: mac/nuc 코어 동기화**

```bash
cd ~/mpcc
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
  /home/hmcl/IFAC2026_SH/src/nonlinear_mpc_acados/ mac/nonlinear_mpc_acados/
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' \
  /home/hmcl/IFAC2026_SH/src/nonlinear_mpc_acados/nonlinear_mpc_acados/ nuc/nonlinear_mpcc/nonlinear_mpcc/
sed -i 's/nonlinear_mpc_acados\.ml/nonlinear_mpcc.ml/g' nuc/nonlinear_mpcc/nonlinear_mpcc/ml/train.py
python3 -m compileall -q mac nuc && echo OK
```

- [ ] **Step 2: README 갱신 + commit + push**

README.md의 소스 시점 문구를 새 커밋 해시로 갱신 후:

```bash
cd ~/mpcc && git add -A && git commit -m "sync: 그립 정직화 (dyn_mu + friction ellipse + 제동 정직화) — IFAC2026_SH@<hash>" && git push
```

- [ ] **Step 3: (T7 연결돼 있으면) T7/mpcc 갱신**

```bash
rsync -a --delete --exclude='.git' ~/mpcc/ /media/hmcl/T7/mpcc/ 2>/dev/null || echo "T7 미연결 — skip"
```

---

### Task 10: BO 게이트 (별도 세션 권장 — 장시간)

- [ ] mu=0.6 BO 재가동 (`scripts/bo_sweep_turbo.py`, BO 시작 전 yaml 스냅샷 + trap 복원 — [[bo-config-clobber-regression]])
- [ ] BO-best **3런 robust 재검증 후에만** 적용 (2026-06-09 교훈)
- [ ] 성공 기준: BO-best 재현 + mu=0.6 랩타임 개선 (현 베이스라인 대비)
- [ ] 결과 커밋 + mpcc 동기화 (Task 9 반복)

---

## Self-Review 체크

- **스펙 커버리지**: §1 dyn_mu 노출=Task 2·3, §2 ellipse=Task 5, §3 ref_v clamp 3지점=Task 4(②③④)·Task 6(build_track), §3b 제동 정직화=Task 4①(bf)·Task 6(a_long), §4 게이트=Task 1·7·8·10. 빠짐 없음.
- **타입/이름 일관성**: `clamp_a_lat_to_grip(a_lat_safe, mu, ellipse_frac)` → (eff, clamped) 튜플 — Task 1 정의·4·6 사용 일치. `A_MIN_DYN=-3.0` Task 1 정의·3(lbu)·6(a_long) 일치. `a_lat_safe_eff()` Task 4 정의·소비 일치.
- **플레이스홀더**: Task 6 Step 1의 ref_v 필드명, Task 3 Step 5의 launch 캐스팅, Task 5의 `u` 심볼명 — 3곳 모두 "구현 시 확인" 지시가 명시돼 있고 확인 방법이 적혀 있음 (실코드가 컨텍스트에 없어 단정 불가한 지점).
- **spec과 차이 (의도)**: ① dyn_mu 주입을 param.get(296-313) 대신 mpc_node attr push로 — use_dynamic/lm_dynamic과 동일한 기존 패턴, 효과 동일 ② codegen mu 키잉은 스펙에 없으나 stale codegen 함정 차단에 필수 ③ clamp를 호출부 3곳 복제 대신 `a_lat_safe_eff()` 단일 게이트로.
