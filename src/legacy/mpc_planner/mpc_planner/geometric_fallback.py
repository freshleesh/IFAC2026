#!/usr/bin/env python3
"""### HJ : Phase 4 Tier-2 GEOMETRIC_FALLBACK.

Frenet quintic으로 (s0, n0, dn_ds0) → (s0+Δs, 0, 0) 복귀 곡선을 만드는
primitive. MPC NLP가 연속 실패할 때 "현재 차 위치에서 raceline 까지의 다리"를
1ms 이내로 깔아주는 역할. 특성:

  - boundary condition 6개 (양끝 n, n', n'') → 5차 다항식 6계수 → 항상 해.
  - 곡률 양끝에서 0 (n''=0)이라 controller lookahead가 덜 놀라며, 최대 곡률도
    (n0, Δs) 로만 결정되므로 예측 가능.
  - s축에 등간격 샘플링 후 raceline 기반 sn_to_xy 로 (x, y, z) 복원.

이 모듈은 Track3D를 쓰지 않고 MPCRacelineLifter 의 인터페이스만 요구한다
(raceline-base 일관성 유지). 샘플 개수는 MPC horizon 과 무관하게 외부에서
지정.
"""

import numpy as np


### HJ : 2026-04-27 v2 — exact port of legacy 3d_recovery_spliner_node.do_spline.
###      User feedback: legacy recovery_spliner is significantly better
###      than my BPoly-only version. Port it faithfully:
###        1) Inflection-based candidate_len (next κ sign change)
###        2) Tangent-by-heading: pick lookahead idx whose direction
###           aligns best with ego yaw
###        3) BPoly cubic Hermite (ego, lookahead) with tangents
###           (ego heading, raceline psi at lookahead) × spline_scale
###        4) Append n_additional=80 GB wpnts after the spline
###        5) Uniform arc-length resampling
###        6) Per-sample (s, n) via signed projection onto raceline
###           tangent-normal at that s (3D-safe, no xy→s projection)
###        7) Stage-1 d-bound validation: |d_arr - center| within
###           [-(d_R - safety), +(d_L - safety)]
###        8) Invalid → return None (caller falls back to next tier)
def build_recovery_path(lifter, s0, n0, ego_x, ego_y, ego_yaw,
                         g_psi=None, g_s=None, g_x=None, g_y=None, g_z=None,
                         g_dleft=None, g_dright=None,
                         g_kappa=None, inflection_points=None,
                         min_candidates_lookahead_n=20, num_kappas=20,
                         max_candidate_len=None,
                         spline_scale=0.8,
                         n_additional=80,
                         wpnt_dist=0.10,
                         delta_s=3.0, n_samples=21,
                         wall_safe=0.15,
                         return_frenet=False):
    """Recovery_spliner-style smooth ego→GB path with wall-aware pull-in.

    Faithful port of legacy 3d_recovery_spliner_node.do_spline.
    Returns (xy_traj (K, 3) [x, y, psi], frenet (K, 2) [s, n]) or
    (None, None) if invalid.
    """
    try:
        from scipy.interpolate import BPoly
    except Exception:
        return (None, None) if return_frenet else None

    # Default wall_safe via lifter accessors if not provided
    if g_dleft is None:
        g_dleft = getattr(lifter, 'g_dleft', None)
    if g_dright is None:
        g_dright = getattr(lifter, 'g_dright', None)
    if g_kappa is None:
        g_kappa = getattr(lifter, 'g_kappa', None)
    g_s_arr = getattr(lifter, 'g_s', None)

    if (g_dleft is None or g_dright is None or g_s_arr is None
            or len(g_s_arr) < 2):
        return (None, None) if return_frenet else None

    ref_max_idx = len(g_s_arr)
    ref_max_s = float(g_s_arr[-1])
    wpnt_dist = float(g_s_arr[1] - g_s_arr[0]) if len(g_s_arr) > 1 else float(wpnt_dist)
    if wpnt_dist <= 0:
        wpnt_dist = 0.1

    cur_s = float(s0)
    cur_s_idx = int(cur_s / wpnt_dist)

    # ---- Inflection-based candidate length ----
    if inflection_points is not None and len(inflection_points) > 0:
        infl_idx_pos = int(np.searchsorted(inflection_points, cur_s_idx))
        if infl_idx_pos == len(inflection_points):
            next_infl = inflection_points[0] + ref_max_idx
        else:
            next_infl = inflection_points[infl_idx_pos]
        candidate_len = int(next_infl - cur_s_idx)
    else:
        candidate_len = ref_max_idx // 2
    candidate_len = int(max(candidate_len, min_candidates_lookahead_n))
    ### HJ : 2026-04-27 — caller-supplied cap on lookahead range. Used by
    ###      shrink retry: if a long endpoint produced wall-violating
    ###      path, caller retries with smaller max_candidate_len so
    ###      tangent_idx is forced to a closer endpoint.
    if max_candidate_len is not None:
        candidate_len = int(min(candidate_len, max(int(max_candidate_len), 5)))
    ### HJ : end

    # gb_idxs: lookahead range
    max_avail = ref_max_idx - 1
    gb_idxs = [(cur_s_idx + i) % ref_max_idx for i in range(candidate_len)]
    gb_idxs = [min(idx, max_avail) for idx in gb_idxs]

    # ---- Tangent-by-heading: pick lookahead idx whose direction
    #     from ego best aligns with raceline psi at that idx ----
    nk = int(min(num_kappas, min_candidates_lookahead_n, len(gb_idxs)))
    xy_m = np.zeros((len(gb_idxs), 2), dtype=np.float64)
    psi_rads = np.zeros(len(gb_idxs), dtype=np.float64)
    for j, idx in enumerate(gb_idxs):
        try:
            x_j, y_j = lifter.sn_to_xy(float(g_s_arr[idx]), 0.0)
            psi_j = lifter._interp_psi(float(g_s_arr[idx]))
        except Exception:
            return (None, None) if return_frenet else None
        xy_m[j, 0] = x_j; xy_m[j, 1] = y_j
        psi_rads[j] = psi_j

    smooth_len = 1.0  # legacy default
    smooth_x = ego_x + np.cos(ego_yaw) * smooth_len
    smooth_y = ego_y + np.sin(ego_yaw) * smooth_len
    dx = xy_m[:, 0] - smooth_x
    dy = xy_m[:, 1] - smooth_y
    norm = np.sqrt(dx * dx + dy * dy) + 1e-9
    unit = np.column_stack([dx / norm, dy / norm])
    psi_unit = np.column_stack([np.cos(psi_rads), np.sin(psi_rads)])
    cos_th = np.clip(np.sum(unit * psi_unit, axis=1), -1.0, 1.0)
    angles = np.arccos(cos_th)
    tangent_idx = int(np.argmin(angles))

    # ---- BPoly cubic Hermite spline (ego, lookahead target) ----
    # Legacy do_spline:
    #   points = [[ego_x, ego_y], [target_x, target_y]]
    #   tangents = unit_vectors @ (spline_scale * eye(2)) = unit_vec * spline_scale
    #   d_cum = [0, L]
    #   spline_result[i, :] = list(zip(ref, tangents[i]))
    #   poly = BPoly.from_derivatives(d_cum, spline_result[:, dim])
    # → endpoint derivative (in arc-length parameter) = unit_vec * spline_scale.
    target_x = xy_m[tangent_idx, 0]
    target_y = xy_m[tangent_idx, 1]
    target_psi = psi_rads[tangent_idx]
    P_start = np.array([ego_x, ego_y], dtype=np.float64)
    P_end = np.array([target_x, target_y], dtype=np.float64)
    L = float(np.linalg.norm(P_end - P_start))
    if L < 1e-3:
        return (None, None) if return_frenet else None

    s_scale = float(spline_scale)
    T_start_x = s_scale * float(np.cos(ego_yaw))
    T_start_y = s_scale * float(np.sin(ego_yaw))
    T_end_x = s_scale * float(np.cos(target_psi))
    T_end_y = s_scale * float(np.sin(target_psi))
    # BPoly.from_derivatives(x, y) where y[i] = [position, slope, ...]
    # at x=x[i]. Legacy uses x = d_cum = [0, L] (arc-length parametrisation).
    d_cum = np.array([0.0, L], dtype=np.float64)
    points_x = np.array([[P_start[0], T_start_x],
                         [P_end[0], T_end_x]], dtype=np.float64)
    points_y = np.array([[P_start[1], T_start_y],
                         [P_end[1], T_end_y]], dtype=np.float64)
    bp_x = BPoly.from_derivatives(d_cum, points_x)
    bp_y = BPoly.from_derivatives(d_cum, points_y)

    # Legacy: nSamples = max(int(l / wpnt_dist), 2); s_param = linspace(0,l,nSamples).
    K_dense = max(int(L / wpnt_dist), 2)
    s_param = np.linspace(0.0, L, K_dense)
    xy_spline = np.column_stack([bp_x(s_param), bp_y(s_param)])

    # ---- Append GB suffix wpnts (legacy n_additional=80) ----
    suffix_idxs = [(tangent_idx + cur_s_idx + i + 1) % ref_max_idx
                   for i in range(int(n_additional))]
    xy_suffix = np.zeros((len(suffix_idxs), 2), dtype=np.float64)
    for j, idx in enumerate(suffix_idxs):
        try:
            xs, ys = lifter.sn_to_xy(float(g_s_arr[idx]), 0.0)
        except Exception:
            return (None, None) if return_frenet else None
        xy_suffix[j, 0] = xs
        xy_suffix[j, 1] = ys
    xy_combined = np.vstack([xy_spline, xy_suffix])

    # ---- Uniform arc-length resampling ----
    diffs = np.diff(xy_combined, axis=0)
    seg = np.sqrt((diffs * diffs).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total_len = float(arc[-1])
    if total_len < 1e-3:
        return (None, None) if return_frenet else None
    n_uni = int(max(int(total_len / wpnt_dist) + 1, int(n_samples)))
    arc_uni = np.linspace(0.0, total_len, n_uni)
    xy = np.column_stack([
        np.interp(arc_uni, arc, xy_combined[:, 0]),
        np.interp(arc_uni, arc, xy_combined[:, 1]),
    ])
    spline_arc_len = float(L)   # legacy: arc[:K_dense-1].cumsum() ≈ L

    # ---- Per-sample (s, n) via signed projection onto raceline normal ----
    s_arr = (cur_s + arc_uni) % ref_max_s
    n_arr = np.zeros(n_uni, dtype=np.float64)
    for i in range(n_uni):
        s_w = float(s_arr[i])
        try:
            x_g, y_g = lifter.sn_to_xy(s_w, 0.0)
            psi_g = lifter._interp_psi(s_w)
        except Exception:
            return (None, None) if return_frenet else None
        nx, ny = -np.sin(psi_g), np.cos(psi_g)
        n_proj = float((xy[i, 0] - x_g) * nx + (xy[i, 1] - y_g) * ny)
        n_arr[i] = n_proj

    # ---- Stage-1 d-bound validation (only spline portion) ----
    n_spline_uni = int(np.searchsorted(arc_uni, spline_arc_len))
    if n_spline_uni >= 2:
        for i in range(min(n_spline_uni, n_uni)):
            s_w = float(s_arr[i])
            d_L = float(lifter._interp(s_w, g_dleft))
            d_R = float(lifter._interp(s_w, g_dright))
            n_lim_up = max(d_L - wall_safe, 0.05)
            n_lim_dn = -max(d_R - wall_safe, 0.05)
            n_i = float(n_arr[i])
            # Skip k=0 (ego's actual position; may be past corridor at spawn)
            if i == 0:
                continue
            if n_i > n_lim_up + 1e-3 or n_i < n_lim_dn - 1e-3:
                return (None, None) if return_frenet else None

    ### HJ : 2026-04-27 — wall-aware pull-in (POST-validation refinement).
    ###      User: "혹여나 recovery 경로가 너무 벽에 붙을때만 붙는 구간
    ###       gb로 당겨서 부드럽게 리샘플링". Only nudge samples that are
    ###      within `wall_pull_thr` of the (already-validated) wall —
    ###      pull them toward GB and savgol-smooth so kinks don't form.
    ###      k=0 always preserved at ego's true position.
    wall_pull_thr = 0.08   # 8cm — start pulling when wall is this close
    wall_pull_alpha = 0.6  # blend strength toward safer position
    pulled_any = False
    for i in range(n_uni):
        if i == 0:
            continue
        s_w = float(s_arr[i])
        d_L = float(lifter._interp(s_w, g_dleft))
        d_R = float(lifter._interp(s_w, g_dright))
        n_lim_up = max(d_L - wall_safe, 0.05)
        n_lim_dn = -max(d_R - wall_safe, 0.05)
        n_i = float(n_arr[i])
        if n_i > n_lim_up - wall_pull_thr:
            n_target = n_lim_up - wall_pull_thr
            n_arr[i] = (1.0 - wall_pull_alpha) * n_i + wall_pull_alpha * n_target
            pulled_any = True
        elif n_i < n_lim_dn + wall_pull_thr:
            n_target = n_lim_dn + wall_pull_thr
            n_arr[i] = (1.0 - wall_pull_alpha) * n_i + wall_pull_alpha * n_target
            pulled_any = True

    if pulled_any:
        try:
            from scipy.signal import savgol_filter
            win = 5
            if n_uni >= win:
                n_arr_orig0 = n_arr[0]
                n_arr = savgol_filter(n_arr, window_length=win, polyorder=2,
                                      mode='nearest')
                n_arr[0] = n_arr_orig0  # preserve ego's true position
        except Exception:
            pass
    ### HJ : end

    # ---- Build output (re-lift xy from final (s, n)) ----
    out = np.zeros((n_uni, 3), dtype=np.float64)
    for i in range(n_uni):
        s_w = float(s_arr[i])
        n_i = float(n_arr[i])
        x, y = lifter.sn_to_xy(s_w, n_i)
        psi = lifter._interp_psi(s_w)
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = psi

    sn = np.column_stack([s_arr, n_arr])
    if return_frenet:
        return out, sn
    return out
### HJ : end


def solve_quintic_coeffs(L, n0, n0_d, n0_dd, n1=0.0, n1_d=0.0, n1_dd=0.0):
    """Solve n(s) = c0..c5 over s∈[0,L] with full clamped BCs.

    Returns the 6-vector [c0, c1, c2, c3, c4, c5] with n(s) = Σ c_i s^i.
    The system is 6×6 and non-singular for L>0, so always returns a solution.
    """
    L = float(max(L, 1e-3))
    L2 = L * L
    L3 = L2 * L
    L4 = L3 * L
    L5 = L4 * L

    # n(0) = c0; n'(0) = c1; n''(0) = 2 c2; so:
    c0 = float(n0)
    c1 = float(n0_d)
    c2 = 0.5 * float(n0_dd)

    # Remaining 3 unknowns c3, c4, c5 from terminal BCs:
    #   n(L)   = c0 + c1 L + c2 L^2 + c3 L^3 + c4 L^4 + c5 L^5   = n1
    #   n'(L)  =        c1   + 2 c2 L + 3 c3 L^2 + 4 c4 L^3 + 5 c5 L^4 = n1_d
    #   n''(L) =               2 c2   + 6 c3 L   + 12 c4 L^2 + 20 c5 L^3 = n1_dd
    A = np.array([
        [L3,      L4,      L5],
        [3*L2,  4*L3,   5*L4],
        [6*L,  12*L2,  20*L3],
    ], dtype=np.float64)
    b = np.array([
        n1    - (c0 + c1 * L + c2 * L2),
        n1_d  - (c1 + 2 * c2 * L),
        n1_dd - (2 * c2),
    ], dtype=np.float64)

    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([c0, c1, c2, c3, c4, c5], dtype=np.float64)


def evaluate_poly(coeffs, s):
    """Horner-style eval of n(s) and n'(s) given 6-coeff vector."""
    c0, c1, c2, c3, c4, c5 = coeffs
    n = c0 + s * (c1 + s * (c2 + s * (c3 + s * (c4 + s * c5))))
    ndot = c1 + s * (2 * c2 + s * (3 * c3 + s * (4 * c4 + s * 5 * c5)))
    return float(n), float(ndot)


def build_quintic_fallback(lifter, s0, n0, psi_delta, delta_s=3.0, n_samples=21,
                           n0_d_hint=None, return_frenet=False):
    """### HJ : Phase 4 Tier-2 entry.

    Parameters
    ----------
    lifter : MPCRacelineLifter
        Provides raceline interpolation + sn_to_xy (raceline-base, matches
        every other consumer downstream of /global_waypoints).
    s0 : float
        Current ego s on raceline. In 3D tracks this MUST come from
        `/car_state/odom_frenet` (z-aware); NEVER from
        `lifter.project_xy_to_sn(car_x, car_y)` which is 2D-only and
        aliases overpass layers.
    n0 : float
        Current ego n offset (from the same 3D Frenet source as s0).
    psi_delta : float
        Ego heading − raceline tangent at s0. Used to set n'(0). Wrapped
        to (-π, π] by caller.
    delta_s : float
        Forward arc length over which to return to the raceline. 8 m ≈ 1.5×
        MPC horizon at 10 m/s with N=20, dT=0.05.
    n_samples : int
        Points to sample along the curve. 21 ≈ MPC horizon + 1.
    n0_d_hint : float, optional
        Override for n'(0). If None, derived from psi_delta via tan(psi_δ).
    return_frenet : bool, default False
        If True, also returns a parallel Frenet (s_world, n) array so the
        node can feed it to `fill_wpnt_from_s` and skip the 2D xy→s round
        trip when publishing Wpnt fields.

    Returns
    -------
    np.ndarray, shape (n_samples, 3) — Cartesian [x, y, psi_tangent].
    If return_frenet:  (xy_traj, np.ndarray shape (n_samples, 2) [s, n]).
    """
    # Small-angle approx ok; explicit tan keeps large psi_delta honest.
    if n0_d_hint is None:
        # Clamp to avoid extreme slopes if ego is pointing 90° off.
        psi_clamped = float(np.clip(psi_delta, -1.0, 1.0))  # ~57°
        n0_d = float(np.tan(psi_clamped))
    else:
        n0_d = float(n0_d_hint)

    coeffs = solve_quintic_coeffs(delta_s, n0, n0_d, 0.0, 0.0, 0.0, 0.0)

    out = np.zeros((n_samples, 3), dtype=np.float64)
    sn = np.zeros((n_samples, 2), dtype=np.float64)
    s_grid = np.linspace(0.0, float(delta_s), n_samples)
    for i, ds in enumerate(s_grid):
        n_i, _ = evaluate_poly(coeffs, ds)
        s_world = s0 + ds
        x, y = lifter.sn_to_xy(s_world, n_i)
        psi = lifter._interp_psi(s_world)
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = psi
        sn[i, 0] = s_world
        sn[i, 1] = n_i
    if return_frenet:
        return out, sn
    return out
