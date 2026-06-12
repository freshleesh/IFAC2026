#!/usr/bin/env python3
"""mintime 라인을 final2_mt 맵 디렉토리로 설치.

- global_waypoints.json: final2 원본 복사 후 global_traj_wpnts_iqp/markers 만 교체
- global_waypoints.csv: 7-col (x,y,w_tr_right,w_tr_left,psi,kappa,vx) closed
- 맵 파일(png/yaml/pbstream 등) 복사 + 이름 변경
"""
import json, csv, math, os, shutil, copy
import numpy as np
from scipy.spatial import cKDTree

WORK = os.path.dirname(os.path.abspath(__file__))
SRC = '/home/hmcl/IFAC2026_SH/src/stack_master/maps/final2'
DST = '/home/hmcl/IFAC2026_SH/src/stack_master/maps/final2_mt'

d = np.load(os.path.join(WORK, 'mintime_traj.npz'))
traj = d['traj_cl']            # closed: [s, x, y, psi(north-0), kappa, vx, ax]
bound_r, bound_l = d['bound_r'], d['bound_l']

# d_right / d_left: 바운드 폴리라인까지 최근접 거리
tree_r, tree_l = cKDTree(bound_r[:, :2]), cKDTree(bound_l[:, :2])
xy = traj[:, 1:3]
d_right, _ = tree_r.query(xy)
d_left, _ = tree_l.query(xy)

# psi_centerline_rad: 최근접 centerline 점의 (변환된) psi — d_eff 보정용
_gj_src = json.load(open(os.path.join(SRC, 'global_waypoints.json')))
_cent = _gj_src['centerline_waypoints']['wpnts']
_cent_xy = np.array([[p['x_m'], p['y_m']] for p in _cent])
_cent_psi = np.array([p['psi_rad'] for p in _cent])
_, _cent_idx = cKDTree(_cent_xy).query(xy)
psi_centerline = _cent_psi[_cent_idx]

def conv_psi(psi):
    new = psi + math.pi / 2
    if new > math.pi:
        new -= 2 * math.pi
    return new

# ── 1) 맵 디렉토리 복사 ──────────────────────────────────────────────
os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    src_f = os.path.join(SRC, f)
    if f.startswith('final2.'):                      # final2.png/yaml/pbstream → final2_mt.*
        dst_f = os.path.join(DST, f.replace('final2.', 'final2_mt.'))
    else:
        dst_f = os.path.join(DST, f)
    shutil.copy2(src_f, dst_f)

# yaml 의 image: 필드 갱신
yaml_p = os.path.join(DST, 'final2_mt.yaml')
txt = open(yaml_p).read().replace('final2.png', 'final2_mt.png')
open(yaml_p, 'w').write(txt)

# ── 2) global_waypoints.json 교체 ───────────────────────────────────
gj = json.load(open(os.path.join(SRC, 'global_waypoints.json')))
wpnt_tmpl = copy.deepcopy(gj['global_traj_wpnts_iqp']['wpnts'][0])
mark_tmpl = copy.deepcopy(gj['global_traj_markers_iqp']['markers'][0])
max_vx = float(traj[:, 5].max())

wpnts, markers = [], []
for i, row in enumerate(traj):
    s, x, y, psi, kap, vx, ax = row
    w = copy.deepcopy(wpnt_tmpl)
    w.update(id=i, s_m=float(s), d_m=0.0, x_m=float(x), y_m=float(y),
             d_right=float(d_right[i]), d_left=float(d_left[i]),
             psi_rad=conv_psi(float(psi)), kappa_radpm=float(kap),
             vx_mps=float(vx), ax_mps2=float(ax),
             psi_centerline_rad=float(psi_centerline[i]))
    wpnts.append(w)
    m = copy.deepcopy(mark_tmpl)
    m['id'] = i
    m['scale']['z'] = float(vx / max_vx)
    m['pose']['position']['x'] = float(x)
    m['pose']['position']['y'] = float(y)
    m['pose']['position']['z'] = float(vx / max_vx / 2)
    markers.append(m)

gj['global_traj_wpnts_iqp']['wpnts'] = wpnts
gj['global_traj_markers_iqp']['markers'] = markers
if isinstance(gj.get('map_info_str'), dict) and 'data' in gj['map_info_str']:
    gj['map_info_str']['data'] += ' [MT] mintime mu0.6 regenerated 2026-06-12, est 16.59s;'
json.dump(gj, open(os.path.join(DST, 'global_waypoints.json'), 'w'))

# ── 3) global_waypoints.csv ─────────────────────────────────────────
with open(os.path.join(DST, 'global_waypoints.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m', 'psi_rad', 'kappa_radpm', 'vx_mps'])
    for i, row in enumerate(traj):
        s, x, y, psi, kap, vx, ax = row
        wr.writerow([f'{x:.6f}', f'{y:.6f}', f'{d_right[i]:.4f}', f'{d_left[i]:.4f}',
                     f'{conv_psi(psi):.6f}', f'{kap:.6f}', f'{vx:.4f}'])

print(f'final2_mt installed: {len(wpnts)} wpnts, L={traj[-1,0]:.2f}m, vx {traj[:,5].min():.2f}-{max_vx:.2f}')
print('files:', sorted(os.listdir(DST)))
