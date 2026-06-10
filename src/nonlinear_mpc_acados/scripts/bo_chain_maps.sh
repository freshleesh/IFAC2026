#!/bin/bash
# 2026-06-02: rand_a BO 끝나면 시험맵 final 에서 BO 이어 실행.
# final = TUM 프로파일(vx 3~11, 긴 직선) 잘 보정됨 → vel_scale=1.0 native + max_speed 11.
# rand_a BO 안 죽임. 각 run_bo.sh 는 자체 trap 으로 yaml 복원 → 순차 안전.
set -u
cd ~/IFAC2026_SH
LOG=/tmp/bo_chain.log
YAML=src/nonlinear_mpc_acados/config/ddrx_unified_params.yaml
echo "[chain] $(date +%H:%M:%S) waiting for rand_a BO to finish..." > "$LOG"
while pgrep -f "bo_sweep_turbo" >/dev/null 2>&1; do sleep 30; done
echo "[chain] $(date +%H:%M:%S) rand_a BO done. preparing final-map config." >> "$LOG"
sleep 5
# final 전용 설정: native 프로파일(vel_scale 1.0) + 높은 cap(11) + 제동 horizon(30)
python3 - <<'PY' >> "$LOG" 2>&1
p="src/nonlinear_mpc_acados/config/ddrx_unified_params.yaml"; s=open(p).read()
import re
s=re.sub(r"(\n    max_speed: )[0-9.]+", r"\g<1>11.0", s, count=1)
s=re.sub(r"(\n    max_speed_p: )[0-9.]+", r"\g<1>11.0", s, count=1)
s=re.sub(r"(\n    vel_scale: )[0-9.]+", r"\g<1>1.0 ", s, count=1)
s=re.sub(r"(\n    N_horizon: )[0-9]+", r"\g<1>30 ", s, count=1)
open(p,"w").write(s); print("[chain] final config: max_speed=11 vel_scale=1.0 N=30")
PY
# trackfinal + maps/final 설치
source /opt/ros/jazzy/setup.bash; source install/setup.bash
rm -rf /tmp/acados_codegen_evompcc* /tmp/acados_ocp_evompcc*
colcon build --packages-select nonlinear_mpc_acados stack_master --symlink-install >> "$LOG" 2>&1
echo "[chain] $(date +%H:%M:%S) starting final-map BO." >> "$LOG"
bash ~/IFAC2026_SH/src/nonlinear_mpc_acados/scripts/run_bo.sh final 40 12 3 160 50 >> "$LOG" 2>&1
echo "[chain] $(date +%H:%M:%S) final-map BO done." >> "$LOG"
