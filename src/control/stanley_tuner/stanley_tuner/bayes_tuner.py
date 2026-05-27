"""섹터별 Gaussian Process Bayesian Optimization.

파라미터 공간: (k, k_ff, lookahead_d) — 3D
목적 함수:    mean(|CTE|) + λ₁·std(steer) + λ₂·mean(|heading_err|)  (최소화)
탐색 전략:    Expected Improvement + 랜덤 탐색 후보 → GP 예측 → EI 최대점

의존성: numpy, scipy만 사용 (scikit-optimize 불필요).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import norm

from stanley_tuner.sector_map import Sector, SectorParams
from stanley_tuner.metrics_collector import SectorStats

N_RANDOM_CANDIDATES = 256   # EI 최대화 후보 수 (512 → 256: 빠르게)
XI = 0.01                   # EI exploration 보정
MIN_OBS_FOR_EI = 5          # EI 탐색 전 최소 관측 수 (이하는 보수적 근방 탐색)


@dataclass
class BayesBounds:
    k_min:   float = 0.5    # 추적 게인 하한 — 이 이하면 경로 추종 불가
    k_max:   float = 5.0
    kff_min: float = 0.0
    kff_max: float = 0.10
    ld_min:  float = 0.3    # lookahead 하한 — 이 이하면 진동/충돌 위험
    ld_max:  float = 3.0


class BayesSectorOptimizer:
    """섹터 하나에 대한 GP-BO 최적화기."""

    def __init__(
        self,
        sector: Sector,
        bounds: BayesBounds | None = None,
        length_scale: float = 0.3,
        sigma_f: float      = 1.0,
        sigma_n: float      = 0.08,
    ):
        self.sector  = sector
        self.bounds  = bounds or BayesBounds()
        self.ls      = length_scale
        self.sf      = sigma_f
        self.sn      = sigma_n

        self._X: List[np.ndarray] = []
        self._y: List[float]      = []
        self._current_x           = self._encode(sector.params)

    # ── public ──────────────────────────────────────────── #

    def observe(self, params: SectorParams, stats: SectorStats) -> None:
        self._X.append(self._encode(params))
        self._y.append(stats.objective)

    def reset(self, sector: Sector) -> None:
        """Safety reset 후 호출 — 관측 데이터 폐기, 안전 파라미터로 재시작."""
        self._X.clear()
        self._y.clear()
        self._current_x = self._encode(sector.params)

    def suggest(self) -> SectorParams:
        """다음 시도할 파라미터 제안."""
        n = len(self._X)
        if n < MIN_OBS_FOR_EI:
            # 관측 부족 → 현재값 주변 소폭 탐색 (sigma 작게 유지)
            sigma = 0.04 + 0.01 * n   # 0.04 ~ 0.08: 점진적으로 넓힘
            return self._random_neighbor(sigma=sigma)

        X = np.array(self._X)
        y = np.array(self._y)

        candidates = np.random.uniform(0.0, 1.0, (N_RANDOM_CANDIDATES, 3))
        ei = self._expected_improvement(candidates, X, y)
        best_idx = int(np.argmax(ei))
        best_x   = candidates[best_idx]

        if ei[best_idx] < 1e-6:
            return self._random_neighbor(sigma=0.06)

        self._current_x = best_x
        return self._decode(best_x)

    def suggest_with_limit(self, current: SectorParams, max_frac: float = 0.1) -> SectorParams:
        """제안값이 현재 파라미터에서 max_frac 이상 벗어나지 않도록 클램핑."""
        raw = self.suggest()
        b   = self.bounds

        def _clamp(v, cur, lo, hi):
            delta = max_frac * cur if cur > 0 else max_frac
            return float(np.clip(v, max(lo, cur - delta), min(hi, cur + delta)))

        return SectorParams(
            k          = round(_clamp(raw.k,          current.k,          b.k_min,  b.k_max),  4),
            k_ff       = round(_clamp(raw.k_ff,       current.k_ff,       b.kff_min, b.kff_max), 6),
            lookahead_d= round(_clamp(raw.lookahead_d, current.lookahead_d if current.lookahead_d > 0
                                      else (b.ld_min + b.ld_max) / 2,     b.ld_min, b.ld_max),  3),
            confidence = min(1.0, len(self._y) * 0.1),
        )

    def best_params(self) -> Optional[SectorParams]:
        if not self._y:
            return None
        best_idx = int(np.argmin(self._y))
        return self._decode(self._X[best_idx])

    def n_observations(self) -> int:
        return len(self._y)

    # ── GP 내부 ─────────────────────────────────────────── #

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff   = X1[:, None, :] - X2[None, :, :]
        sqdist = np.sum(diff ** 2, axis=-1) / (self.ls ** 2)
        return self.sf ** 2 * np.exp(-0.5 * sqdist)

    def _gp_predict(self, X_cand, X_obs, y_obs):
        n        = len(X_obs)
        K        = self._kernel(X_obs, X_obs) + (self.sn ** 2 + 1e-6) * np.eye(n)
        K_star   = self._kernel(X_cand, X_obs)
        K_ss_diag = self.sf ** 2 * np.ones(len(X_cand))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-3 * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
        mu    = K_star @ alpha
        v     = np.linalg.solve(L, K_star.T)
        var   = K_ss_diag - np.sum(v ** 2, axis=0)
        sigma = np.sqrt(np.maximum(var, 1e-10))
        return mu, sigma

    def _expected_improvement(self, X_cand, X_obs, y_obs) -> np.ndarray:
        mu, sigma = self._gp_predict(X_cand, X_obs, y_obs)
        f_best = float(np.min(y_obs))
        z  = (f_best - mu - XI) / sigma
        ei = (f_best - mu - XI) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-10] = 0.0
        return ei

    # ── 파라미터 인코딩/디코딩 ──────────────────────────── #

    def _encode(self, p: SectorParams) -> np.ndarray:
        b = self.bounds
        def _norm(v, lo, hi): return np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        ld = p.lookahead_d if p.lookahead_d > 0 else (b.ld_min + b.ld_max) / 2
        return np.array([
            _norm(p.k,   b.k_min,   b.k_max),
            _norm(p.k_ff, b.kff_min, b.kff_max),
            _norm(ld,    b.ld_min,  b.ld_max),
        ], dtype=float)

    def _decode(self, x: np.ndarray) -> SectorParams:
        b = self.bounds
        x = np.clip(x, 0.0, 1.0)
        def _denorm(v, lo, hi): return v * (hi - lo) + lo
        return SectorParams(
            k          = round(float(_denorm(x[0], b.k_min,   b.k_max)),  4),
            k_ff       = round(float(_denorm(x[1], b.kff_min, b.kff_max)), 6),
            lookahead_d= round(float(_denorm(x[2], b.ld_min,  b.ld_max)),  3),
            confidence = min(1.0, self.n_observations() * 0.1),
        )

    def _random_neighbor(self, sigma: float = 0.06) -> SectorParams:
        x = self._current_x + np.random.normal(0, sigma, 3)
        x = np.clip(x, 0.0, 1.0)
        return self._decode(x)


# ── 섹터 전체 관리 ────────────────────────────────────── #

class BayesTunerManager:
    def __init__(self, sectors: List[Sector], bounds: BayesBounds | None = None, **kwargs):
        self._bounds    = bounds or BayesBounds()
        self._sectors   = {s.id: s for s in sectors}
        self._opts: Dict[int, BayesSectorOptimizer] = {
            s.id: BayesSectorOptimizer(s, bounds=self._bounds, **kwargs)
            for s in sectors
        }

    def observe(self, sector_id: int, params: SectorParams, stats: SectorStats) -> None:
        if sector_id in self._opts:
            self._opts[sector_id].observe(params, stats)

    def suggest(self, sector_id: int, current: SectorParams) -> Optional[SectorParams]:
        if sector_id not in self._opts:
            return None
        return self._opts[sector_id].suggest_with_limit(current)

    def reset(self, sector_id: int) -> None:
        """Safety reset 시 호출 — 해당 섹터 Bayes 상태 초기화."""
        if sector_id in self._opts and sector_id in self._sectors:
            self._opts[sector_id].reset(self._sectors[sector_id])

    def best_params(self, sector_id: int) -> Optional[SectorParams]:
        if sector_id not in self._opts:
            return None
        return self._opts[sector_id].best_params()

    def n_obs(self, sector_id: int) -> int:
        return self._opts[sector_id].n_observations() if sector_id in self._opts else 0
