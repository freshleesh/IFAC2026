#!/bin/bash
# auto_speed_collect.sh — C 학습 데이터 자동 수집.
# 4 m/s 부터 12 m/s 까지 1 m/s 씩 증가. 각 단계마다 정확히 N lap.
#
# 동작:
#   1. yaml 의 max_speed/max_speed_p 를 sed 로 갱신
#   2. launch 백그라운드 실행
#   3. /mpc/lap_count 토픽 폴링 → N lap 도달하면 launch 종료
#   4. 다음 max_speed 로 반복
#
# 데이터: ~/mpc_logs/mpc_*.csv 자동 누적.

set -e
cd "$(dirname "$0")/../../.."  # → /home/hmcl/IFAC2026_SH

YAML="src/nonlinear_mpc_acados/config/ddrx_unified_params.yaml"
N_LAPS=5                  # 단계당 lap 수
POLL_INTERVAL=3           # 초마다 lap_count 확인
INIT_TIMEOUT=120          # init 단계 (codegen + auto_engage) 안에 첫 lap 안 시작하면 abort

echo "============================="
echo "  Auto-speed data collection"
echo "  4.0 → 6.0 m/s @ 0.5 step, $N_LAPS lap/step (lap_count 기반)"
echo "  더 많은 데이터를 위해 0.5 m/s 간격 — 5 단계."
echo "============================="

for v in 4.0 4.5 5.0 5.5 6.0; do
    echo
    echo "============================="
    echo "  step: max_speed = $v m/s"
    echo "============================="

    sed -i.bak "s/^    max_speed:.*/    max_speed: $v/" "$YAML"
    sed -i.bak "s/^    max_speed_p:.*/    max_speed_p: $v/" "$YAML"
    grep -E "^    max_speed(:|_p:)" "$YAML"

    # launch 백그라운드
    ros2 launch stack_master full_sim.launch.py mode:=mpcc \
        > "/tmp/mpcc_step_${v}.log" 2>&1 &
    LAUNCH_PID=$!
    echo "  launch pid: $LAUNCH_PID"

    # /mpc/lap_count 폴링
    START=$(date +%s)
    LAST_LAP=-1
    LAST_LAP_TIME=$START   # 마지막으로 lap 이 늘어난 시각 — 박혀서 멈춤 탐지용
    STUCK_TIMEOUT=60       # lap 진행 0 으로 이만큼 지나면 박힌 걸로 보고 abort
    while true; do
        sleep "$POLL_INTERVAL"
        NOW=$(date +%s)
        ELAPSED=$(( NOW - START ))

        # latched 토픽이라 --once 면 즉시 반환
        LAP=$(timeout 2 ros2 topic echo --once /mpc/lap_count 2>/dev/null \
              | awk '/^data:/ {print $2; exit}')
        if [ -z "$LAP" ]; then LAP=0; fi

        if [ "$LAP" != "$LAST_LAP" ]; then
            echo "  [${ELAPSED}s] lap = $LAP"
            LAST_LAP=$LAP
            LAST_LAP_TIME=$NOW
        fi

        if [ "$LAP" -ge "$N_LAPS" ]; then
            echo "  → reached $N_LAPS lap, stopping launch"
            break
        fi
        if [ "$ELAPSED" -gt "$INIT_TIMEOUT" ] && [ "$LAP" -lt 1 ]; then
            echo "  → init timeout ($INIT_TIMEOUT s) without any lap — aborting step"
            break
        fi
        # 박힘 탐지: lap 진행 0 으로 STUCK_TIMEOUT 초 지나면 abort
        NO_PROGRESS=$(( NOW - LAST_LAP_TIME ))
        if [ "$LAP" -ge 1 ] && [ "$NO_PROGRESS" -gt "$STUCK_TIMEOUT" ]; then
            echo "  → stuck (no lap progress for ${STUCK_TIMEOUT}s at lap=$LAP) — aborting step"
            break
        fi
    done

    # launch 종료 — graceful 먼저, 그래도 살아있으면 SIGKILL.
    kill -INT -- -"$LAUNCH_PID" 2>/dev/null || kill -INT "$LAUNCH_PID" 2>/dev/null || true
    sleep 3
    kill -KILL -- -"$LAUNCH_PID" 2>/dev/null || kill -KILL "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true

    # 모든 자식 ROS 프로세스 강제 정리 (이름 기반 — group kill 이 놓친 zombie 잡기)
    pkill -KILL -f "gym_bridge|state_machine|spliner|controller_manager|global_republisher|frenet_conversion|frenet_odom_republisher|static_obstacle_manager|fake_topic_relay|simple_mux|ego_robot_state_publisher|rviz2|mpc_node|mpc_debug_logger" 2>/dev/null
    # ros2 daemon 도 도메인 stale state 안 남도록 정리
    pkill -KILL -f "ros2.*daemon" 2>/dev/null
    sleep 1

    LATEST_CSV=$(ls -t ~/mpc_logs/mpc_*.csv 2>/dev/null | head -1)
    if [ -n "$LATEST_CSV" ]; then
        ROWS=$(wc -l < "$LATEST_CSV")
        echo "  → $LATEST_CSV: $ROWS rows"
    fi

    sleep 3   # DDS / TCP 소켓 / shared memory 정리 시간 — 다음 launch 가 깨끗하게 시작하도록
done

echo
echo "============================="
echo "  Done. CSV files:"
ls -la ~/mpc_logs/mpc_*.csv | tail -10
echo "  Total rows: $(wc -l ~/mpc_logs/mpc_*.csv | tail -1 | awk '{print $1}')"
echo "============================="
echo "  Next: train MLP"
echo "    conda activate dgm"
echo "    python3 -m nonlinear_mpc_acados.ml.train --epochs 200"
echo "============================="
