"""State enum — 원본 그대로 (rospy 의존 없음)."""
import enum


class StateType(enum.Enum):
    GB_TRACK = "GB_TRACK"
    TRAILING = "TRAILING"
    OVERTAKE = "OVERTAKE"
    FTGONLY = "FTGONLY"
    RECOVERY = "RECOVERY"
    ATTACK = "ATTACK"
    START = "START"
    LOSTLINE = "LOSTLINE"
    SMART_STATIC = "SMART_STATIC"
