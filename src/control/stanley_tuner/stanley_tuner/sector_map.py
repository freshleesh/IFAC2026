"""섹터 맵: 코너 감지, 데이터 구조, YAML 저장/로드."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np
import yaml


class CornerType(str, Enum):
    STRAIGHT      = "straight"
    SHALLOW       = "shallow_corner"   # |kappa| 0.15–0.35 rad/m
    MEDIUM        = "medium_corner"    # |kappa| 0.35–0.65 rad/m
    DEEP          = "deep_corner"      # |kappa| > 0.65 rad/m


def _corner_type(kappa_max: float) -> CornerType:
    k = abs(kappa_max)
    if k < 0.15:
        return CornerType.STRAIGHT
    elif k < 0.35:
        return CornerType.SHALLOW
    elif k < 0.65:
        return CornerType.MEDIUM
    else:
        return CornerType.DEEP


@dataclass
class SectorParams:
    k:          float = 2.5
    k_ff:       float = 0.001
    lookahead_d: float = 0.0   # 0 = stanley의 adaptive 공식 유지
    confidence: float = 0.0    # 0–1: 학습된 데이터 양 (랩 수 기반)


@dataclass
class Sector:
    id:          int
    start_s:     float
    end_s:       float
    kappa_mean:  float
    kappa_max:   float
    corner_type: CornerType
    params:      SectorParams = field(default_factory=SectorParams)

    @property
    def length(self) -> float:
        return self.end_s - self.start_s

    def contains(self, s: float, track_length: float) -> bool:
        """s가 이 섹터 안에 있는지 (랩 경계 wrap-around 처리)."""
        if self.start_s <= self.end_s:
            return self.start_s <= s < self.end_s
        # wrap-around 섹터 (트랙 끝 → 처음 걸치는 경우)
        return s >= self.start_s or s < self.end_s


# ────────────────────────────────────────────────────────────── #
# 자동 섹터 감지
# ────────────────────────────────────────────────────────────── #

def detect_sectors(
    s_arr: np.ndarray,
    kappa_arr: np.ndarray,
    kappa_threshold: float = 0.15,
    min_sector_len: float  = 1.0,
    smooth_window:  int    = 5,
) -> List[Sector]:
    """global_waypoints의 s, kappa 배열에서 섹터 목록을 자동 생성.

    Args:
        s_arr:           누적 arc-length [m] (N,)
        kappa_arr:       곡률 [rad/m] (N,)
        kappa_threshold: 코너 판정 임계값 [rad/m]
        min_sector_len:  최소 섹터 길이 [m] (너무 짧은 구간 병합)
        smooth_window:   kappa 이동평균 윈도우 크기

    Returns:
        Sector 리스트 (직선 + 코너 구간 교대 배치)
    """
    N = len(s_arr)
    # kappa 이동평균 스무딩
    kernel = np.ones(smooth_window) / smooth_window
    kappa_smooth = np.convolve(np.abs(kappa_arr), kernel, mode='same')

    # 코너 여부 (1 = corner, 0 = straight)
    is_corner = (kappa_smooth >= kappa_threshold).astype(int)

    # run-length 인코딩으로 구간 추출
    changes = np.diff(is_corner, prepend=-1, append=-1)
    starts  = np.where(changes != 0)[0]

    raw_segments = []
    for i in range(len(starts) - 1):
        si = starts[i]
        ei = starts[i + 1]
        seg_s_start = float(s_arr[si])
        seg_s_end   = float(s_arr[min(ei, N - 1)])
        seg_is_corner = bool(is_corner[si])
        kappa_seg = kappa_smooth[si:ei]
        raw_segments.append({
            "start_s":    seg_s_start,
            "end_s":      seg_s_end,
            "is_corner":  seg_is_corner,
            "kappa_mean": float(np.mean(kappa_seg)),
            "kappa_max":  float(np.max(kappa_seg)),
        })

    # 짧은 구간 인접 구간으로 병합
    merged = _merge_short(raw_segments, min_sector_len)

    # Sector 객체 생성
    sectors: List[Sector] = []
    for i, seg in enumerate(merged):
        ctype = _corner_type(seg["kappa_max"]) if seg["is_corner"] else CornerType.STRAIGHT
        sectors.append(Sector(
            id=i,
            start_s=seg["start_s"],
            end_s=seg["end_s"],
            kappa_mean=seg["kappa_mean"],
            kappa_max=seg["kappa_max"],
            corner_type=ctype,
        ))

    return sectors


def _merge_short(segs: list, min_len: float) -> list:
    """min_len보다 짧은 구간을 인접 구간에 병합."""
    if not segs:
        return segs
    changed = True
    while changed:
        changed = False
        result = []
        i = 0
        while i < len(segs):
            seg = segs[i]
            if (seg["end_s"] - seg["start_s"]) < min_len and len(segs) > 1:
                # 앞 구간에 병합 (없으면 뒤 구간)
                if result:
                    prev = result[-1]
                    prev["end_s"]      = seg["end_s"]
                    prev["kappa_mean"] = (prev["kappa_mean"] + seg["kappa_mean"]) / 2
                    prev["kappa_max"]  = max(prev["kappa_max"], seg["kappa_max"])
                    prev["is_corner"]  = prev["is_corner"] or seg["is_corner"]
                elif i + 1 < len(segs):
                    nxt = segs[i + 1]
                    nxt["start_s"]    = seg["start_s"]
                    nxt["kappa_mean"] = (nxt["kappa_mean"] + seg["kappa_mean"]) / 2
                    nxt["kappa_max"]  = max(nxt["kappa_max"], seg["kappa_max"])
                    nxt["is_corner"]  = nxt["is_corner"] or seg["is_corner"]
                    i += 1
                    result.append(segs[i])
                    i += 1
                    continue
                changed = True
                i += 1
                continue
            result.append(seg)
            i += 1
        segs = result
    return segs


# ────────────────────────────────────────────────────────────── #
# YAML 저장 / 로드
# ────────────────────────────────────────────────────────────── #

def save_yaml(path: str, sectors: List[Sector], defaults: SectorParams) -> None:
    import os as _os
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    data = {
        "defaults": {
            "k":          defaults.k,
            "k_ff":       defaults.k_ff,
            "lookahead_d": defaults.lookahead_d,
        },
        "sectors": [
            {
                "id":          s.id,
                "start_s":     round(s.start_s, 3),
                "end_s":       round(s.end_s, 3),
                "type":        s.corner_type.value,
                "kappa_mean":  round(s.kappa_mean, 4),
                "kappa_max":   round(s.kappa_max, 4),
                "k":           round(s.params.k, 4),
                "k_ff":        round(s.params.k_ff, 6),
                "lookahead_d": round(s.params.lookahead_d, 3),
                "confidence":  round(s.params.confidence, 3),
            }
            for s in sectors
        ],
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: str) -> tuple[List[Sector], SectorParams]:
    """YAML에서 섹터 목록과 기본값 로드. 파일 없으면 ([], None) 반환."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return [], SectorParams()

    defs = data.get("defaults", {})
    defaults = SectorParams(
        k=defs.get("k", 2.5),
        k_ff=defs.get("k_ff", 0.01),
        lookahead_d=defs.get("lookahead_d", 0.0),
    )
    sectors = []
    for d in data.get("sectors", []):
        p = SectorParams(
            k=d.get("k", defaults.k),
            k_ff=d.get("k_ff", defaults.k_ff),
            lookahead_d=d.get("lookahead_d", defaults.lookahead_d),
            confidence=d.get("confidence", 0.0),
        )
        sectors.append(Sector(
            id=d["id"],
            start_s=d["start_s"],
            end_s=d["end_s"],
            kappa_mean=d.get("kappa_mean", 0.0),
            kappa_max=d.get("kappa_max", 0.0),
            corner_type=CornerType(d.get("type", "straight")),
            params=p,
        ))
    return sectors, defaults


def apply_yaml_override(
    auto_sectors: List[Sector],
    yaml_sectors: List[Sector],
) -> List[Sector]:
    """자동 감지 섹터에 YAML override 적용. id 매칭으로 params 덮어쓰기."""
    yaml_by_id = {s.id: s for s in yaml_sectors}
    for sec in auto_sectors:
        if sec.id in yaml_by_id:
            sec.params = yaml_by_id[sec.id].params
    return auto_sectors


def find_sector(sectors: List[Sector], s: float, track_length: float) -> Optional[Sector]:
    """현재 s 위치에 해당하는 섹터 반환."""
    s_norm = s % track_length
    for sec in sectors:
        if sec.contains(s_norm, track_length):
            return sec
    # fallback: 가장 가까운 섹터
    if sectors:
        return min(sectors, key=lambda sec: abs(sec.start_s - s_norm))
    return None
