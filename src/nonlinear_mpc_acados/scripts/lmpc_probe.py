#!/usr/bin/env python3
"""(a) step 1 — probe the EXISTING softmin LMPC at N=25 + raceline seed.

Runs a long sim (many laps) with use_lmpc=true (set in yaml), then prints the
PER-LAP time progression so we can see whether LMPC improves lap-by-lap
(Rosolia-monotonic) or drifts up (the known softmin failure). before-data for
the hard-constraint surgery decision.
"""
import csv
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from bo_sweep_turbo import run_one_sim  # noqa: E402

MAP = 'rand_a'
N_LAPS = 16
WALL = 400
STUCK = 45

print(f'=== LMPC probe: {MAP} {N_LAPS} laps @ N=25 (use_lmpc=true, raceline seed) ===', flush=True)
res = run_one_sim(MAP, N_LAPS, WALL, STUCK)
print('reason:', res.get('reason'), ' n_resets:', res.get('n_resets'), flush=True)
csv_path = res.get('csv')
print('csv:', csv_path, flush=True)

if not (csv_path and Path(csv_path).exists()):
    print('NO CSV', flush=True); sys.exit(0)

rows = list(csv.DictReader(open(csv_path)))
# per-lap time from lap-column transitions
laps = {}
for r in rows:
    try:
        L = int(float(r['lap'])); t = float(r['t'])
    except Exception:
        continue
    laps.setdefault(L, []).append(t)
keys = sorted(laps)
print('\n=== per-lap times ===', flush=True)
prev_end = None
for L in keys:
    t0, t1 = laps[L][0], laps[L][-1]
    dur = t1 - t0
    print(f'  lap {L:>2}: {dur:5.2f}s  (n={len(laps[L])})', flush=True)
# also LMPC active fraction if column exists
if rows and 'lap' in rows[0]:
    pass
print('=== lmpc_probe done ===', flush=True)
