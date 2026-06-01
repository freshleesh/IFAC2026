#!/usr/bin/env python3
"""(b) EVO small-N limit sweep — find the N_horizon floor at fixed BO-best weights.

Tests the EVO hypothesis directly: with strong q_v (the q_v-aware BO best already
applied in the live yaml), how small can N go before corners wedge / lap time
degrades? Sweeps N over a list, ONE sim per N (one-sim-at-a-time), scoring with
the hardened eval_run_quality. Only N_horizon is changed between runs — weights
are left exactly as the BO set them. Original N restored at the end.

Run ONLY after the BO finishes (never concurrently — single-sim rule).

Usage:
  python3 nsweep_horizon.py                 # default N list 35,30,25,20
  python3 nsweep_horizon.py 40 35 30 25 20  # custom
"""
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from bo_sweep_turbo import run_one_sim  # noqa: E402

YAML = Path.home() / 'IFAC2026_SH/src/nonlinear_mpc_acados/config/ddrx_unified_params.yaml'
EVAL = SCRIPTS / 'eval_run_quality.py'
MAP = 'rand_a'
N_LAPS = 6
WALL = 220
STUCK = 45

N_LIST = [int(a) for a in sys.argv[1:]] or [35, 30, 25, 20]


def set_n(n: int) -> None:
    txt = YAML.read_text()
    new = re.sub(r'(^\s*N_horizon:\s*)\d+', rf'\g<1>{n}', txt, count=1, flags=re.M)
    assert new != txt or f'N_horizon: {n}' in txt, 'N_horizon line not found/changed'
    YAML.write_text(new)


def cur_n() -> int:
    m = re.search(r'^\s*N_horizon:\s*(\d+)', YAML.read_text(), flags=re.M)
    return int(m.group(1)) if m else -1


orig_n = cur_n()
print(f'=== N-sweep @ BO-best weights (orig N={orig_n}) — list {N_LIST} ===', flush=True)
rows = []
try:
    for n in N_LIST:
        set_n(n)
        # force acados regen for the new horizon
        subprocess.run('rm -rf /tmp/acados_codegen_evompcc /tmp/acados_ocp_evompcc.json',
                       shell=True, check=False)
        print(f'\n--- N={n} ---', flush=True)
        res = run_one_sim(MAP, N_LAPS, WALL, STUCK)
        csv = res.get('csv')
        row = {'N': n, 'reason': res.get('reason'), 'n_resets': res.get('n_resets', 0)}
        if csv and Path(csv).exists():
            ev = subprocess.run(
                ['python3', str(EVAL), '--csv', csv, '--map', MAP, '--json',
                 '--n_resets', str(res.get('n_resets', 0))],
                capture_output=True, text=True)
            try:
                m = json.loads(ev.stdout)
                row.update({
                    'lap': round(m.get('lap_time_mean', 0), 2),
                    'Q': round(m.get('Q', 0), 1),
                    'success': m.get('success'),
                    'STUCK': round(m.get('stuck_frac', 0), 4),
                    'lat_g_peak': round(m.get('lat_g_peak', 0), 1),
                    'shake': round(m.get('shake_rms', 0), 3),
                    'solve_ms': round(m.get('solve_ms_mean', 0), 2),
                    'fail': m.get('fail_reasons'),
                })
            except Exception as e:
                row['eval_err'] = f'{e}: {ev.stdout[:200]} {ev.stderr[:200]}'
        else:
            row['eval_err'] = 'no csv'
        rows.append(row)
        print('  ->', json.dumps(row, default=str), flush=True)
finally:
    set_n(orig_n)
    print(f'\n=== restored N_horizon={orig_n} ===', flush=True)

print('\n=== SUMMARY ===', flush=True)
print(f"{'N':>4} {'lap':>7} {'STUCK':>7} {'lat_gpk':>8} {'shake':>6} {'solve':>6} {'ok':>5}", flush=True)
for r in rows:
    print(f"{r['N']:>4} {r.get('lap','-'):>7} {r.get('STUCK','-'):>7} "
          f"{r.get('lat_g_peak','-'):>8} {r.get('shake','-'):>6} "
          f"{r.get('solve_ms','-'):>6} {str(r.get('success','-')):>5}", flush=True)
print('=== nsweep done ===', flush=True)
