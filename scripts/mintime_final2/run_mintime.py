#!/usr/bin/env python3
"""final2 mintime raceline 재생성 (mu=0.6) — 스탠드얼론 드라이버.

gb_optimizer_25d.trajectory_optimizer(curv_opt_type='mintime') 를 ROS 없이 직접 호출.
입력: final2 global_waypoints.json 의 centerline (x, y, d_right, d_left)
출력: mintime_traj.npz (trajectory[s,x,y,psi,kappa,vx,ax], bound_r, bound_l, est_t)
"""
import sys, os, json
import numpy as np

WORK = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK)

# 의존 경로: GRO(TUMFTM 원본, car-ws) + spliner(iqp wrapper, IFAC ws install)
sys.path.insert(0, '/home/hmcl/creating_autonomous_car_ws/src/creating_autonomous_car/planner')
sys.path.insert(0, '/home/hmcl/IFAC2026_SH/install/spliner/lib/python3.12/site-packages/spliner')
sys.path.insert(0, '/home/hmcl/IFAC2026_SH/install/gb_optimizer_25d/lib/python3.12/site-packages')
sys.path.insert(0, '/home/hmcl/IFAC2026_SH/install/vel_planner_25d/lib/python3.12/site-packages')

# 1) centerline CSV 생성 (x_m, y_m, w_tr_right_m, w_tr_left_m)
src_json = '/home/hmcl/IFAC2026_SH/src/stack_master/maps/final2/global_waypoints.json'
d = json.load(open(src_json))
w = d['centerline_waypoints']['wpnts']
cent = np.array([[p['x_m'], p['y_m'], p['d_right'], p['d_left']] for p in w])
np.savetxt('map_centerline.csv', cent, delimiter=',',
           header='x_m,y_m,w_tr_right_m,w_tr_left_m', comments='# ')
print(f'centerline: {len(cent)} pts written')

# tph spline_approximation 이 scipy.euclidean 에 (2,1) 배열을 넘기는 버전 비호환 → ravel 패치
import scipy.spatial.distance as _sd
_orig_euclidean = _sd.euclidean
_sd.euclidean = lambda u, v, w=None: _orig_euclidean(np.ravel(u), np.ravel(v), w)

# 2) mintime 최적화 (opt_mintime 은 DM→float 패치 사본으로 교체 주입)
# traj_opt_patched: reopt_mintime_solution=True (mincurv 재최적화로 kappa<=curvlim 강제)
#                   recalc_vel_profile_by_tph=True (mu0.6 ggv 기반 속도 재계산)
import importlib.util
_spec_t = importlib.util.spec_from_file_location('traj_opt_patched',
                                                 os.path.join(WORK, 'traj_opt_patched.py'))
_mod_t = importlib.util.module_from_spec(_spec_t)
_spec_t.loader.exec_module(_mod_t)
trajectory_optimizer = _mod_t.trajectory_optimizer
import global_racetrajectory_optimization.opt_mintime_traj.src.opt_mintime as _om
_spec = importlib.util.spec_from_file_location('opt_mintime_patched',
                                               os.path.join(WORK, 'opt_mintime_patched.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_om.opt_mintime = _mod.opt_mintime

traj_cl, bound_r, bound_l, est_t, z_fine, slope, z_bounds = trajectory_optimizer(
    input_path=WORK,
    track_name='map_centerline',
    curv_opt_type='mintime',
    safety_width=0.8,
    plot=False)

print(f'\n=== mintime done: est lap time {est_t:.3f}s ===')
print(f'traj shape {traj_cl.shape}, vx range {traj_cl[:,5].min():.2f}-{traj_cl[:,5].max():.2f} m/s')

np.savez(os.path.join(WORK, 'mintime_traj.npz'),
         traj_cl=traj_cl, bound_r=bound_r, bound_l=bound_l,
         est_t=est_t)
print('saved mintime_traj.npz')
