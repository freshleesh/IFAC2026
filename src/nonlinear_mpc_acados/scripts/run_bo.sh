#!/bin/bash
# run_bo.sh - BO 학습 wrapper. ros2 launch bo_train.launch.xml 에서 호출.
#
# Usage (direct):
#   bash run_bo.sh rand_a 20 5 3 180 60 "" alternate bucketed
#
# Args (positional):
#   1 map           single map name (--map). 무시 if 7번째 maps != ""
#   2 n_calls       BO total iterations
#   3 n_initial     Sobol initial samples
#   4 n_laps        trial 당 lap
#   5 wall_timeout  sim wall timeout [s]
#   6 stuck_timeout stuck detect timeout [s]
#   7 maps          multi-map (공백 구분). "" 이면 single --map 사용
#   8 map_mode      alternate | mean
#   9 mode          bucketed (only)

set -e

MAP=${1:-rand_a}
N_CALLS=${2:-20}
N_INITIAL=${3:-5}
N_LAPS=${4:-3}
WALL_TIMEOUT=${5:-180}
STUCK_TIMEOUT=${6:-60}
MAPS=${7:-}
MAP_MODE=${8:-alternate}
MODE=${9:-bucketed}
X0=${10:-}     # warm-start scale CSV (e.g. "4.0,2.0,1.0,1.5,1.0,3.0,1.5")

# pkill leftover sim/mpc.
# NOTE: 패턴에서 "ros2.*launch" 와 "nonlinear_mpc" 제외 — 우리 부모 ros2 launch
# 의 cmdline 에 "nonlinear_mpc_acados" 포함되어 자기 죽임 ("Killed"). 구체 node
# 이름만 (mpc_node, mpc_debug) 매칭.
pkill -9 -f "gym_bridge|mpc_node|mpc_debug|state_machine|frenet|simple_mux|rviz2|fake_topic|pp_fallback|ftg_fallback|global_republisher|joy_node|robot_state|obstacle" 2>/dev/null || true
sleep 5

cd ~/IFAC2026_SH
source /opt/ros/jazzy/setup.bash
source install/local_setup.bash
export CYCLONEDDS_URI="file://$HOME/cyclonedds.xml"

mkdir -p ~/bo_results
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$HOME/bo_results/bo_${MAP}_${TS}.log"

# ── 재발방지 (2026-05-29 leftover-weight 회귀) ──────────────────────────
# bo_sweep_turbo.py 의 sed_yaml_override() 가 매 trial 마다 ddrx_unified_params.yaml
# 의 q_*_scale_live 를 in-place 로 덮어씀. 중간에 kill 되면 마지막 (나쁜) trial
# weight 가 그대로 남아 다음 실행/실차 base 를 망침 (실제로 발생: shake 0.32,
# STUCK 50). 종료(정상/INT/TERM) 시 항상 pre-BO 스냅샷으로 복원 — BO 의 best 는
# bo_turbo_*.json 에 있으니 거기서 의도적으로 적용한다.
BO_YAML="$HOME/IFAC2026_SH/src/nonlinear_mpc_acados/config/ddrx_unified_params.yaml"
BO_YAML_BACKUP="$HOME/bo_results/ddrx_prebo_backup_${TS}.yaml"
cp "$BO_YAML" "$BO_YAML_BACKUP"
trap 'cp "$BO_YAML_BACKUP" "$BO_YAML"; echo "[run_bo] live ddrx yaml restored to pre-BO config (leftover-weight 방지)"' EXIT INT TERM

if [ -z "$MAPS" ]; then
    MAP_ARG=( --map "$MAP" )
else
    MAP_ARG=( --maps "$MAPS" --map_mode "$MAP_MODE" )
fi

X0_ARG=()
if [ -n "$X0" ]; then
    X0_ARG=( --x0 "$X0" )
fi

echo "============================="
echo " BO sweep start: map=$MAP  n_calls=$N_CALLS  n_initial=$N_INITIAL  n_laps=$N_LAPS"
echo " log -> $LOGFILE"
echo "============================="

python3 -u src/nonlinear_mpc_acados/scripts/bo_sweep_turbo.py \
    --mode "$MODE" \
    --n_calls "$N_CALLS" --n_initial "$N_INITIAL" --n_laps "$N_LAPS" \
    "${MAP_ARG[@]}" \
    --wall_timeout "$WALL_TIMEOUT" --stuck_timeout "$STUCK_TIMEOUT" \
    "${X0_ARG[@]}" \
    2>&1 | tee "$LOGFILE"
