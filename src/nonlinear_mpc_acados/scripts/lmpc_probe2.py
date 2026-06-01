#!/usr/bin/env python3
"""Robust LMPC probe — FIXED-duration run + CSV analysis (no lap_count polling).

The /mpc/lap_count poll goes stale (a prior run's latched value trips an instant
early-exit). This probe instead: aggressive cleanup → launch full_sim → run a
fixed wall-time → kill → parse the freshly-written mpc CSV for per-lap times +
max |e_c|. Reports the lap-by-lap trend so we can see drift / wall-clip / speed.
"""
import csv
import glob
import os
import signal
import subprocess
import time
import numpy as np

MAP = 'rand_a'
RUN_SEC = 340          # ~16 laps at ~18-20s
HOME = os.path.expanduser('~')

KILL = ("pkill -9 -f 'gym_bridge|mpc_node|full_sim|ros2 launch|simple_mux|"
        "rviz2|global_republisher|frenet|spliner|state_machine|fake_topic|obstacle|"
        "joy_node|robot_state' 2>/dev/null; pkill -9 -f '_ros2_daemon|ros2.*daemon' 2>/dev/null; "
        "ros2 daemon stop 2>/dev/null; true")


def sh(cmd):
    return subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)


print('=== robust LMPC probe (fixed-duration, CSV-based) ===', flush=True)
print('cleanup...', flush=True)
sh(KILL); time.sleep(5)

# mark time so we only pick up the NEW csv
t_launch = time.time()
env = os.environ.copy()
env['CYCLONEDDS_URI'] = f'file://{HOME}/cyclonedds.xml'
cmd = ('source /opt/ros/jazzy/setup.bash && source ~/IFAC2026_SH/install/local_setup.bash && '
       f'export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml && '
       f'ros2 launch stack_master full_sim.launch.py mode:=mpcc map:={MAP}')
logf = open('/tmp/lmpc_probe2_sim.log', 'w')
proc = subprocess.Popen(['bash', '-c', cmd], stdout=logf, stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid, env=env)
print(f'launched, running {RUN_SEC}s...', flush=True)
try:
    for i in range(RUN_SEC // 10):
        time.sleep(10)
        if proc.poll() is not None:
            print('sim exited early', flush=True); break
finally:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    sh(KILL); time.sleep(2)

# newest mpc csv created after launch
csvs = sorted(glob.glob(f'{HOME}/mpc_logs/mpc_*.csv'), key=os.path.getmtime)
csv_path = csvs[-1] if csvs else None
print('csv:', csv_path, flush=True)
if not csv_path or os.path.getmtime(csv_path) < t_launch - 5:
    print('NO fresh CSV — sim failed to start/log', flush=True)
    raise SystemExit

rows = list(csv.DictReader(open(csv_path)))
lap = np.array([int(float(r['lap'])) for r in rows])
t = np.array([float(r['t']) for r in rows])
ec = np.array([abs(float(r.get('e_c', r.get('cte', 0)) or 0)) for r in rows]) if ('e_c' in rows[0] or 'cte' in rows[0]) else None
vmax = max(float(r['v_actual']) for r in rows)
tr = np.where(np.diff(lap) > 0)[0]
print(f'\ntotal laps={lap.max()}  rows={len(rows)}  v_actual_max={vmax:.2f}', flush=True)
if len(tr) >= 2:
    lt = np.diff(t[tr])
    print('=== per-lap times ===', flush=True)
    for i, x in enumerate(lt):
        seg = ''
        if ec is not None:
            a, b = tr[i], tr[i + 1]
            seg = f'  max|ec|={ec[a:b].max():.2f}'
        print(f'  lap {i+1}: {x:5.2f}s{seg}', flush=True)
    print(f'\nmedian(settled lap2+)={np.median(lt[1:]):.2f}  min={lt.min():.2f}', flush=True)
else:
    print(f'only {len(tr)} lap transition(s) — ran {t[-1]-t[0]:.0f}s', flush=True)
print('=== probe2 done ===', flush=True)
