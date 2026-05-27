"""섹터별 주행 메트릭 수집.

한 바퀴 동안 각 섹터를 통과할 때마다 CTE / heading_error / steer 샘플을 모으고,
섹터를 빠져나가는 시점에 해당 구간의 통계를 계산해 반환한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from stanley_tuner.sector_map import Sector


@dataclass
class SectorStats:
    sector_id:   int
    cte_mean:    float   # mean(|CTE|)
    steer_std:   float   # std(steer)  — oscillation proxy
    hdg_mean:    float   # mean(|heading_error|)
    n_samples:   int
    objective:   float   # λ 가중 합산 (낮을수록 좋음)


class MetricsCollector:
    """50 Hz 루프에서 호출되며 섹터별 샘플을 누적한다."""

    def __init__(
        self,
        lambda1: float = 0.3,   # steer_std 가중치
        lambda2: float = 0.2,   # hdg_mean 가중치
    ):
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self._cur_sector_id: Optional[int] = None
        self._cte_buf:   List[float] = []
        self._steer_buf: List[float] = []
        self._hdg_buf:   List[float] = []

        # 섹터별 완료된 랩 통계 히스토리
        self._history: Dict[int, List[SectorStats]] = {}

    # ── public ──────────────────────────────────────────── #

    def update(
        self,
        sector: Optional[Sector],
        cte:   float,
        steer: float,
        hdg:   float,
    ) -> Optional[SectorStats]:
        """50 Hz 마다 호출. 섹터가 바뀌는 순간 완료된 섹터 통계를 반환."""
        new_id = sector.id if sector is not None else None

        if new_id != self._cur_sector_id:
            # 섹터 전환 → 이전 섹터 통계 확정
            finished = self._flush()
            self._cur_sector_id = new_id
            self._cte_buf   = []
            self._steer_buf = []
            self._hdg_buf   = []
            return finished

        # 현재 섹터 샘플 누적
        if sector is not None:
            self._cte_buf.append(abs(cte))
            self._steer_buf.append(steer)
            self._hdg_buf.append(abs(hdg))

        return None

    def history(self, sector_id: int) -> List[SectorStats]:
        return self._history.get(sector_id, [])

    def lap_count(self, sector_id: int) -> int:
        return len(self._history.get(sector_id, []))

    def clear_history(self, sector_id: int) -> None:
        self._history.pop(sector_id, None)

    # ── private ─────────────────────────────────────────── #

    def _flush(self) -> Optional[SectorStats]:
        if self._cur_sector_id is None or len(self._cte_buf) < 5:
            return None

        cte_arr   = np.array(self._cte_buf)
        steer_arr = np.array(self._steer_buf)
        hdg_arr   = np.array(self._hdg_buf)

        cte_mean  = float(np.mean(cte_arr))
        steer_std = float(np.std(steer_arr))
        hdg_mean  = float(np.mean(hdg_arr))
        obj = cte_mean + self.lambda1 * steer_std + self.lambda2 * hdg_mean

        stats = SectorStats(
            sector_id=self._cur_sector_id,
            cte_mean=cte_mean,
            steer_std=steer_std,
            hdg_mean=hdg_mean,
            n_samples=len(cte_arr),
            objective=obj,
        )
        self._history.setdefault(self._cur_sector_id, []).append(stats)
        return stats
