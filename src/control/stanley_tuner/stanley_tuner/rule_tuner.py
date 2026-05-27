"""룰 기반 파라미터 적응.

각 섹터를 통과한 후 CTE / oscillation / heading_error 통계에 따라
k, k_ff, lookahead_d를 조정한다.

룰:
  1. CTE ↑   → k ↑  (추적 게인 증가)
  2. CTE ↓↓  + osc ↓ → k 소폭 감소 (과도 게인 완화)
  3. osc ↑   → k ↓  (진동 억제)
  4. hdg ↑   + 코너  → k_ff ↑ (피드포워드 강화)
  5. lookahead_d: 코너 곡률에 반비례 (더 타이트한 코너 → 더 짧은 lookahead)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from stanley_tuner.sector_map import Sector, SectorParams, CornerType
from stanley_tuner.metrics_collector import SectorStats


@dataclass
class RuleConfig:
    """rule_tuner 하이퍼파라미터. tuner_params.yaml에서 로드된다."""
    # 파라미터 조정 배율
    k_alpha:       float = 0.07   # k 증가 배율 (CTE 높을 때)
    k_beta:        float = 0.12   # k 감소 배율 (진동 있을 때)
    kff_alpha:     float = 0.10   # k_ff 조정 배율

    # 트리거 임계값
    cte_high:      float = 0.08   # [m]   k 증가 트리거
    cte_low:       float = 0.03   # [m]   k 감소 트리거
    osc_high:      float = 0.04   # [rad] 진동 판정 steer_std
    hdg_high:      float = 0.08   # [rad] k_ff 증가 트리거

    # lookahead 계산
    base_lookahead: float = 1.5   # [s]  기준 시간
    kappa_scale:    float = 1.8   # 곡률 영향 스케일

    # 파라미터 클램프 범위
    k_min:   float = 0.5
    k_max:   float = 10.0
    kff_min: float = 0.0
    kff_max: float = 0.15
    ld_min:  float = 0.3
    ld_max:  float = 5.0


def suggest(
    sector: Sector,
    stats_list: List[SectorStats],
    min_laps: int = 2,
    cfg: RuleConfig | None = None,
) -> SectorParams:
    """최근 min_laps 개 통계를 평균해 새 파라미터 제안.

    데이터가 부족하면 현재 파라미터를 그대로 반환.
    """
    if cfg is None:
        cfg = RuleConfig()

    if len(stats_list) < min_laps:
        return sector.params

    recent = stats_list[-min_laps:]
    cte_mean  = np.mean([s.cte_mean  for s in recent])
    steer_std = np.mean([s.steer_std for s in recent])
    hdg_mean  = np.mean([s.hdg_mean  for s in recent])

    p = sector.params
    k    = p.k
    k_ff = p.k_ff
    ld   = p.lookahead_d

    # ── 룰 1/2/3: k 조정 ──────────────────────── #
    if steer_std > cfg.osc_high:
        # 룰 3: 진동 우선 — k 감소
        k = max(k * (1.0 - cfg.k_beta), cfg.k_min)
    elif cte_mean > cfg.cte_high:
        # 룰 1: CTE 크면 k 증가
        k = min(k * (1.0 + cfg.k_alpha), cfg.k_max)
    elif cte_mean < cfg.cte_low and steer_std < cfg.osc_high * 0.5:
        # 룰 2: 충분히 좋으면 k 소폭 감소 (과도 게인 완화)
        k = max(k * (1.0 - cfg.k_alpha * 0.4), cfg.k_min)

    # ── 룰 4: k_ff 조정 (코너에서만) ──────────── #
    is_corner = sector.corner_type != CornerType.STRAIGHT
    if is_corner:
        if hdg_mean > cfg.hdg_high:
            k_ff = min(k_ff * (1.0 + cfg.kff_alpha), cfg.kff_max)
        elif hdg_mean < cfg.hdg_high * 0.3 and k_ff > 0.005:
            k_ff = max(k_ff * (1.0 - cfg.kff_alpha * 0.5), cfg.kff_min)

    # ── 룰 5: lookahead_d (기하학 기반) ──────── #
    kappa_eff = abs(sector.kappa_max)
    ld_new = cfg.base_lookahead / (1.0 + cfg.kappa_scale * kappa_eff)
    ld_new = float(np.clip(ld_new, cfg.ld_min, cfg.ld_max))

    # 기존 ld와 EMA 블렌딩 (급격한 변화 방지)
    ld = 0.7 * ld + 0.3 * ld_new if ld > 0 else ld_new

    new_conf = min(1.0, p.confidence + 0.15)

    return SectorParams(k=round(k, 4), k_ff=round(k_ff, 6),
                        lookahead_d=round(ld, 3), confidence=new_conf)
