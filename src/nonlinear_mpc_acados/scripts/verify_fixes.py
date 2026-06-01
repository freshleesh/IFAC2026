#!/usr/bin/env python3
"""One-shot verification after the 2026-06-01 audit fix batch.

Launches ONE sim with the current (deploy) ddrx config — NO weight override —
drives N laps, then scores with the hardened eval_run_quality (lat_g_max=15).
Confirms: (1) 17.84s deploy holds, (2) STUCK/resets = 0, (3) lat_g_max=15 does
NOT false-trigger now real vy/r is used, (4) acados log shows a_lat hard cap=10.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from bo_sweep_turbo import run_one_sim  # noqa: E402

MAP = 'rand_a'
N_LAPS = 7
WALL = 220
STUCK = 45

print(f'=== verify: {MAP}  n_laps={N_LAPS} (deploy config, no override) ===', flush=True)
res = run_one_sim(MAP, N_LAPS, WALL, STUCK)
print('\n--- run_one_sim result ---', flush=True)
print(json.dumps({k: v for k, v in res.items() if k != 'csv'}, indent=2, default=str), flush=True)
csv = res.get('csv')
print('csv:', csv, flush=True)

# grep the launch log for the a_lat hard-cap line + any STUCK
log = '/tmp/bo_trial_sim.log'
if Path(log).exists():
    txt = Path(log).read_text(errors='ignore')
    m = re.search(r'a_lat hard cap = ([\d.]+)', txt)
    print('a_lat hard cap log:', m.group(0) if m else '(NOT FOUND)', flush=True)
    print('STUCK lines:', txt.count('stuck-recover'), flush=True)

# score with hardened objective (defaults now lat_g_max=15, cte_max=0.8, reset_ceiling=2)
if csv and Path(csv).exists():
    print('\n--- eval_run_quality (hardened defaults) ---', flush=True)
    ev = subprocess.run(
        ['python3', str(SCRIPTS / 'eval_run_quality.py'), '--csv', csv,
         '--map', MAP, '--json', '--n_resets', str(res.get('n_resets', 0))],
        capture_output=True, text=True)
    print(ev.stdout, flush=True)
    if ev.returncode != 0:
        print('eval stderr:', ev.stderr, flush=True)
else:
    print('NO CSV — sim failed to log', flush=True)
print('=== verify done ===', flush=True)
