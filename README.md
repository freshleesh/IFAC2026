# IFAC2026 — ROS2 Jazzy Migration

`ICRA2026_HJ` (ROS1 Noetic) 코드베이스의 **state_machine + 관련 노드** 를 ROS2 Jazzy 로
단계적으로 포팅한 워크스페이스. SH 단독 작업.

자세한 분석 / 계획 / 진행 트래킹은 [MIGRATION.md](MIGRATION.md) 참조.

---

## 빌드

```bash
cd ~/unicorn_ws/ICRA2026_SH_ros2
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

## 워크스페이스 구조

```
ICRA2026_SH_ros2/
├── src/
│   └── f110_msgs/         ← Phase A-1: 메시지 (18개)
└── MIGRATION.md           ← 분석 + 단계별 계획
```

---

## 외부 의존 (Skip)

cartographer / GLIM / livox / vesc / blink — **이번 마이그레이션에서 제외**.
검증에 필수가 아니며 별도 ROS2 포팅 작업이 필요.

## 원본 ROS1 워크스페이스

`~/unicorn_ws/ICRA2026_HJ/` — 변경 없이 그대로 유지.
