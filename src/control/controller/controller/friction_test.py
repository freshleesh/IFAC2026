#!/usr/bin/env python3
"""
friction_test.py — F1TENTH 자동 마찰 계수 측정 노드

테스트 순서 (기본값):
  1. accel       : 직선 최대 가속 → ax_max
  2. brake       : 직선 최대 제동 → a_brake_max
  3. circle_ccw  : 좌회전 원주행 속도 스윕 → ay_max
  4. circle_cw   : 우회전 원주행 속도 스윕 → ay_max (검증)

각 테스트 전 countdown_s 동안 차량 재배치 가능.

필요 공간:
  직선 테스트 : 20m 이상 직선
  원주행 테스트: 직경 4m 이상 개방 공간

Usage:
  ros2 run controller friction_test
  ros2 run controller friction_test --ros-args \\
    -p odom_topic:=/car_state/odom \\
    -p circle_steering_deg:=20.0 \\
    -p v_max_test:=5.0 \\
    -p countdown_s:=10

특정 테스트만
  ros2 run controller friction_test --ros-args \
    -p test_sequence:="['circle_ccw', 'circle_cw']"


출력:
  ~/friction_test_<timestamp>.csv  (시계열 로그)
  터미널 최종 요약 + friction_circle.yaml / ggv.csv 권장값
"""

import csv
import math
import time
from collections import deque
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry


# ── 상태 정의 ─────────────────────────────────────────────────────────────────

class S(IntEnum):
    INIT          = 0
    COUNTDOWN     = 1
    ACCEL_RUN     = 10   # 최대 스로틀 → a_long 측정
    ACCEL_STOP    = 11   # 정지 대기
    BRAKE_SPINUP  = 20   # v_target_brake 도달 대기
    BRAKE_MEAS    = 21   # 스로틀 0 → a_long 측정
    BRAKE_WAIT    = 22   # 정지 대기
    CIRCLE_RAMP   = 30   # 목표 속도까지 가속
    CIRCLE_HOLD   = 31   # 속도 유지, a_lat 측정 + 슬립 감지
    CIRCLE_STOP   = 32   # 정지 대기
    DONE          = 99


# 각 상태 안전 타임아웃 [s]
_TIMEOUT = {
    S.COUNTDOWN:   120.0,
    S.ACCEL_RUN:    10.0,
    S.ACCEL_STOP:   10.0,
    S.BRAKE_SPINUP: 20.0,
    S.BRAKE_MEAS:   10.0,
    S.BRAKE_WAIT:   10.0,
    S.CIRCLE_RAMP:  10.0,
    S.CIRCLE_HOLD:  30.0,
    S.CIRCLE_STOP:  10.0,
}


# ── 노드 ─────────────────────────────────────────────────────────────────────

class FrictionTest(Node):

    # ── 초기화 ───────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('friction_test')

        # 파라미터 선언
        self.declare_parameter('odom_topic',          '/car_state/odom')
        self.declare_parameter('drive_topic',         '/vesc/high_level/ackermann_cmd')
        self.declare_parameter('control_rate_hz',      50.0)
        self.declare_parameter('output_dir',           str(Path.home()))
        self.declare_parameter('countdown_s',          10)
        self.declare_parameter('test_sequence',        ['accel', 'brake',
                                                        'circle_ccw', 'circle_cw'])
        # 직선 테스트
        self.declare_parameter('v_max_test',           5.0)   # [m/s] 안전 상한
        self.declare_parameter('v_target_brake',       4.0)   # [m/s] 제동 시작 속도
        self.declare_parameter('accel_duration_s',     3.0)   # [s]  가속 측정 시간
        # 원주행 테스트
        self.declare_parameter('circle_steering_deg', 20.0)   # [°] 고정 조향각
        self.declare_parameter('v_start_circle',       1.0)   # [m/s] 시작 속도
        self.declare_parameter('v_step_circle',        0.3)   # [m/s] 속도 증분
        self.declare_parameter('hold_time_s',          3.0)   # [s]  각 스텝 유지 시간
        # 슬립 감지
        self.declare_parameter('slip_std_threshold',   2.0)   # [m/s²] std > 이 값 → 슬립
        self.declare_parameter('slip_drop_ratio',      0.85)  # a_lat < 이전×비율 → 슬립
        self.declare_parameter('kappa_drop_ratio',     0.85)  # 곡률 κ 감소 비율 → 반경 팽창 슬립

        def p(n): return self.get_parameter(n).value

        odom_topic  = str(p('odom_topic'))
        drive_topic = str(p('drive_topic'))
        self._dt    = 1.0 / float(p('control_rate_hz'))
        output_dir  = Path(str(p('output_dir'))).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        self._countdown_s     = int(p('countdown_s'))
        self._test_seq: List[str] = list(p('test_sequence'))
        self._v_max           = float(p('v_max_test'))
        self._v_tgt_brake     = float(p('v_target_brake'))
        self._accel_dur       = float(p('accel_duration_s'))
        self._steer_rad       = math.radians(float(p('circle_steering_deg')))
        self._v_start_circle  = float(p('v_start_circle'))
        self._v_step          = float(p('v_step_circle'))
        self._hold_time       = float(p('hold_time_s'))
        self._slip_std_thr    = float(p('slip_std_threshold'))
        self._slip_drop_ratio = float(p('slip_drop_ratio'))
        self._kappa_drop_ratio = float(p('kappa_drop_ratio'))

        # CSV 초기화
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._csv_path = output_dir / f'friction_test_{stamp}.csv'
        self._file   = open(self._csv_path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            't_s', 'test', 'state',
            'vx', 'omega_z',
            'a_lat', 'a_long',
            'cmd_speed', 'cmd_steer_deg',
        ])

        # odom 상태
        self._odom: Optional[Odometry] = None
        self._prev_vx:     Optional[float] = None
        self._prev_odom_t: Optional[float] = None
        self._vx_buf: deque = deque(maxlen=5)

        # 측정 버퍼
        self._a_long_samples: List[float] = []
        self._a_lat_per_step: List[float] = []   # 원주행 각 스텝의 평균 |a_lat|
        self._hold_buf: deque = deque(maxlen=500)
        self._kappa_buf: deque = deque(maxlen=500)  # 곡률 κ = |a_lat|/vx² 버퍼
        self._kappa_step_ref: Optional[float] = None  # 현 스텝 초반 기준 곡률

        # 결과
        self._results: dict = {}

        # 상태머신
        self._state      = S.INIT
        self._state_t    = 0.0
        self._state_enter_t = time.time()
        self._test_idx   = -1
        self._circle_v   = self._v_start_circle
        self._circle_sign = 1.0    # +1=CCW, -1=CW
        self._countdown_last_tick = -1
        self._t0 = time.time()

        # ROS
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self._pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.create_timer(self._dt, self._loop)

        self.get_logger().info(
            f'[FrictionTest] 준비 완료\n'
            f'  테스트 순서 : {self._test_seq}\n'
            f'  v_max       : {self._v_max} m/s\n'
            f'  원주행 조향 : {math.degrees(self._steer_rad):.1f}°\n'
            f'  CSV         : {self._csv_path}\n'
            f'  중단: Ctrl+C (비상 정지 자동 실행)'
        )

    # ── odom 콜백 ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self._odom = msg

    # ── 메인 루프 ─────────────────────────────────────────────────────────────

    def _loop(self):
        if self._odom is None:
            return

        odom    = self._odom
        vx      = float(odom.twist.twist.linear.x)
        omega_z = float(odom.twist.twist.angular.z)
        odom_t  = (odom.header.stamp.sec
                   + odom.header.stamp.nanosec * 1e-9)

        # a_long: 속도 미분 + 이동평균
        a_long = 0.0
        if self._prev_vx is not None and self._prev_odom_t is not None:
            dt_o = odom_t - self._prev_odom_t
            if dt_o > 1e-4:
                self._vx_buf.append((vx - self._prev_vx) / dt_o)
        self._prev_vx     = vx
        self._prev_odom_t = odom_t
        if self._vx_buf:
            a_long = float(np.mean(self._vx_buf))

        # a_lat: 원심 가속도 vx·ω_z
        a_lat = vx * omega_z

        # 상태머신 (wall-clock 기반 경과 시간)
        self._state_t = time.time() - self._state_enter_t
        cmd_speed, cmd_steer = self._step(vx, a_long, a_lat)

        # drive 퍼블리시
        msg = AckermannDriveStamped()
        msg.header.stamp         = self.get_clock().now().to_msg()
        msg.drive.speed          = float(cmd_speed)
        msg.drive.steering_angle = float(cmd_steer)
        self._pub.publish(msg)

        # CSV 기록
        if self._file.closed:
            return
        elapsed   = time.time() - self._t0
        test_name = (self._test_seq[self._test_idx]
                     if 0 <= self._test_idx < len(self._test_seq) else 'none')
        self._writer.writerow([
            f'{elapsed:.4f}', test_name, int(self._state),
            f'{vx:.4f}', f'{omega_z:.4f}',
            f'{a_lat:.4f}', f'{a_long:.4f}',
            f'{cmd_speed:.4f}', f'{math.degrees(cmd_steer):.2f}',
        ])

    # ── 상태머신 ──────────────────────────────────────────────────────────────

    def _step(self, vx: float, a_long: float, a_lat: float):
        """(cmd_speed [m/s], cmd_steer [rad]) 반환."""

        # ── INIT ──────────────────────────────────────────────────────────
        if self._state == S.INIT:
            self._next_test()
            return 0.0, 0.0

        # ── COUNTDOWN ─────────────────────────────────────────────────────
        if self._state == S.COUNTDOWN:
            tick = int(self._state_t)
            rem  = self._countdown_s - tick
            if tick != self._countdown_last_tick:
                self._countdown_last_tick = tick
                name = self._test_seq[self._test_idx]
                if rem > 0:
                    self.get_logger().info(
                        f'[{name}] 시작까지 {rem:2d}초 — '
                        f'차량을 테스트 위치에 배치하세요')
                else:
                    self.get_logger().info(f'[{name}] ▶ 시작!')
                    self._begin_test()
            return 0.0, 0.0

        # ─────────────────── 직선 가속 ────────────────────────────────────

        if self._state == S.ACCEL_RUN:
            if self._state_t < 0.1:             # 시작 직후 1회 로그
                self.get_logger().info(
                    f'[accel] RUN 시작 — v_max={self._v_max} m/s, '
                    f'dur={self._accel_dur} s')
            if vx > 0.3:                        # 정지 직후 노이즈 제외
                self._a_long_samples.append(a_long)
            done = (self._state_t >= self._accel_dur
                    or vx >= self._v_max * 0.98
                    or self._state_t > _TIMEOUT[S.ACCEL_RUN])
            if done:
                reason = ('duration'     if self._state_t >= self._accel_dur else
                          'speed_limit'  if vx >= self._v_max * 0.98 else 'timeout')
                self.get_logger().info(
                    f'[accel] 종료: {reason}  '
                    f'state_t={self._state_t:.2f}s  vx={vx:.2f} m/s  '
                    f'samples={len(self._a_long_samples)}')
                if self._a_long_samples:
                    ax = max(float(np.percentile(self._a_long_samples, 90)), 0.0)
                    self._results['accel'] = {'ax_max': ax}
                    self.get_logger().info(
                        f'[accel] ax_max = {ax:.2f} m/s²  '
                        f'(p90, n={len(self._a_long_samples)})')
                else:
                    self.get_logger().warn(
                        '[accel] 샘플 없음 — 차량이 움직이지 않았거나 '
                        '/drive 명령이 전달되지 않았습니다.')
                self._set_state(S.ACCEL_STOP)
            return self._v_max, 0.0

        if self._state == S.ACCEL_STOP:
            if vx < 0.15 or self._state_t > _TIMEOUT[S.ACCEL_STOP]:
                self._next_test()
            return 0.0, 0.0

        # ─────────────────── 직선 제동 ────────────────────────────────────

        if self._state == S.BRAKE_SPINUP:
            if (vx >= self._v_tgt_brake * 0.95
                    or self._state_t > _TIMEOUT[S.BRAKE_SPINUP]):
                if vx < self._v_tgt_brake * 0.5:
                    self.get_logger().warn(
                        f'[brake] 목표 {self._v_tgt_brake:.1f} m/s 미달 '
                        f'({vx:.1f} m/s) — 그래도 제동 시작')
                self._a_long_samples.clear()
                self._set_state(S.BRAKE_MEAS)
            return self._v_tgt_brake, 0.0

        if self._state == S.BRAKE_MEAS:
            self._a_long_samples.append(a_long)
            if (vx < 0.1
                    or self._state_t > _TIMEOUT[S.BRAKE_MEAS]):
                if self._a_long_samples:
                    ab = abs(float(np.percentile(self._a_long_samples, 10)))
                    self._results['brake'] = {'a_brake_max': ab}
                    self.get_logger().info(
                        f'[brake] a_brake_max = {ab:.2f} m/s²  '
                        f'(|p10|, n={len(self._a_long_samples)})')
                self._set_state(S.BRAKE_WAIT)
            return 0.0, 0.0

        if self._state == S.BRAKE_WAIT:
            if vx < 0.05 or self._state_t > _TIMEOUT[S.BRAKE_WAIT]:
                self._next_test()
            return 0.0, 0.0

        # ─────────────────── 원주행 스윕 ──────────────────────────────────

        if self._state == S.CIRCLE_RAMP:
            steer = self._circle_sign * self._steer_rad
            on_speed = abs(vx - self._circle_v) < 0.15
            timeout  = self._state_t > _TIMEOUT[S.CIRCLE_RAMP]
            if on_speed or timeout:
                if timeout and not on_speed:
                    self.get_logger().warn(
                        f'[circle] v={self._circle_v:.1f} 미달 '
                        f'({vx:.1f} m/s) — 그래도 유지 시작')
                self._hold_buf.clear()
                self._kappa_buf.clear()
                self._kappa_step_ref = None
                self._set_state(S.CIRCLE_HOLD)
            return self._circle_v, steer

        if self._state == S.CIRCLE_HOLD:
            steer  = self._circle_sign * self._steer_rad
            test_k = 'circle_ccw' if self._circle_sign > 0 else 'circle_cw'
            self._hold_buf.append(abs(a_lat))

            # 곡률 κ = |a_lat| / vx²  (= |ω_z| / vx = 1/R)
            vx_safe = max(abs(vx), 0.3)
            kappa = abs(a_lat) / (vx_safe ** 2)
            self._kappa_buf.append(kappa)

            # 기준 곡률: HOLD 진입 후 처음 20샘플(0.4s)로 설정
            if self._kappa_step_ref is None and len(self._kappa_buf) >= 20:
                self._kappa_step_ref = float(np.mean(list(self._kappa_buf)[:20]))

            # ── 슬립 감지 ──────────────────────────────────────────────
            slip = False
            if len(self._hold_buf) >= 30:
                recent = list(self._hold_buf)[-30:]
                std  = float(np.std(recent))
                mean = float(np.mean(recent))
                # 기준 1: a_lat 불안정 (표준편차 급등)
                if std > self._slip_std_thr:
                    slip = True
                    self.get_logger().warn(
                        f'[{test_k}] 슬립 감지 (a_lat std={std:.2f}) '
                        f'v={self._circle_v:.1f} m/s')
                # 기준 2: a_lat이 이전 스텝 대비 감소 (그립 손실)
                if (not slip
                        and self._a_lat_per_step
                        and mean < self._a_lat_per_step[-1] * self._slip_drop_ratio):
                    slip = True
                    self.get_logger().warn(
                        f'[{test_k}] 슬립 감지 (a_lat 감소) '
                        f'v={self._circle_v:.1f} m/s')
                # 기준 3: 곡률 감소 → 반경 팽창 (언더스티어/슬립)
                if (not slip
                        and self._kappa_step_ref is not None
                        and len(self._kappa_buf) >= 30):
                    recent_kappa = float(np.mean(list(self._kappa_buf)[-30:]))
                    if recent_kappa < self._kappa_step_ref * self._kappa_drop_ratio:
                        slip = True
                        self.get_logger().warn(
                            f'[{test_k}] 슬립 감지 (반경 팽창: '
                            f'κ {self._kappa_step_ref:.3f}→{recent_kappa:.3f}, '
                            f'R {1/max(self._kappa_step_ref,1e-3):.2f}→'
                            f'{1/max(recent_kappa,1e-3):.2f} m) '
                            f'v={self._circle_v:.1f} m/s')

            done = (slip
                    or self._state_t >= self._hold_time
                    or self._state_t > _TIMEOUT[S.CIRCLE_HOLD])

            if done:
                # 이번 스텝 평균 기록
                if self._hold_buf:
                    step_lat = float(np.mean(list(self._hold_buf)))
                    self._a_lat_per_step.append(step_lat)
                    self.get_logger().info(
                        f'[{test_k}] v={self._circle_v:.1f} m/s  '
                        f'a_lat={step_lat:.2f} m/s²'
                        + ('  ← 슬립!' if slip else ''))

                if slip:
                    # 슬립 직전 스텝까지의 최대값을 ay_max로 채택
                    pre = self._a_lat_per_step[:-1]
                    ay = float(max(pre)) if pre else float(self._a_lat_per_step[0])
                    self._results[test_k] = {'ay_max': ay}
                    self.get_logger().info(
                        f'[{test_k}] ▶ ay_max = {ay:.2f} m/s²  (슬립 직전)')
                    self._set_state(S.CIRCLE_STOP)
                    return 0.0, 0.0

                # 다음 속도 스텝으로
                next_v = self._circle_v + self._v_step
                if next_v > self._v_max:
                    ay = float(max(self._a_lat_per_step)) \
                         if self._a_lat_per_step else 0.0
                    self._results[test_k] = {'ay_max': ay}
                    self.get_logger().info(
                        f'[{test_k}] 스윕 완료. ay_max = {ay:.2f} m/s²')
                    self._set_state(S.CIRCLE_STOP)
                    return 0.0, 0.0
                else:
                    self._circle_v = next_v
                    self._hold_buf.clear()
                    self._kappa_buf.clear()
                    self._kappa_step_ref = None
                    self._set_state(S.CIRCLE_RAMP)
                    return self._circle_v, steer

            return self._circle_v, steer

        if self._state == S.CIRCLE_STOP:
            if vx < 0.15 or self._state_t > _TIMEOUT[S.CIRCLE_STOP]:
                self._next_test()
            return 0.0, 0.0

        # ── DONE ─────────────────────────────────────────────────────────
        return 0.0, 0.0

    # ── 헬퍼 ─────────────────────────────────────────────────────────────────

    def _set_state(self, state: S):
        self._state        = state
        self._state_t      = 0.0
        self._state_enter_t = time.time()

    def _next_test(self):
        """다음 테스트로 진행 (카운트다운 경유)."""
        self._test_idx += 1
        if self._test_idx >= len(self._test_seq):
            self._set_state(S.DONE)
            self._print_summary()
            return
        # 측정 버퍼 초기화
        self._a_long_samples.clear()
        self._a_lat_per_step = []
        self._hold_buf.clear()
        self._kappa_buf.clear()
        self._kappa_step_ref = None
        self._circle_v = self._v_start_circle
        self._countdown_last_tick = -1
        self._set_state(S.COUNTDOWN)
        self.get_logger().info(
            f'──────── 다음 테스트: '
            f'{self._test_seq[self._test_idx]} ────────')

    def _begin_test(self):
        """카운트다운 완료 후 첫 번째 활성 상태 진입."""
        self._a_long_samples.clear()
        test = self._test_seq[self._test_idx]
        if test == 'accel':
            self._set_state(S.ACCEL_RUN)
        elif test == 'brake':
            self._set_state(S.BRAKE_SPINUP)
        elif test == 'circle_ccw':
            self._circle_sign = 1.0
            self._circle_v    = self._v_start_circle
            self._a_lat_per_step = []
            self._set_state(S.CIRCLE_RAMP)
        elif test == 'circle_cw':
            self._circle_sign = -1.0
            self._circle_v    = self._v_start_circle
            self._a_lat_per_step = []
            self._set_state(S.CIRCLE_RAMP)
        else:
            self.get_logger().warn(f'알 수 없는 테스트: {test}, 건너뜀')
            self._next_test()

    # ── 최종 요약 ─────────────────────────────────────────────────────────────

    def _print_summary(self):
        self._file.flush()
        self._file.close()

        ax  = self._results.get('accel',      {}).get('ax_max',      None)
        ab  = self._results.get('brake',       {}).get('a_brake_max', None)
        ccw = self._results.get('circle_ccw',  {}).get('ay_max',      None)
        cw  = self._results.get('circle_cw',   {}).get('ay_max',      None)

        ay_vals = [v for v in [ccw, cw] if v is not None]
        ay_min  = float(min(ay_vals))  if ay_vals else None
        ay_mean = float(np.mean(ay_vals)) if ay_vals else None

        g = 9.81
        sep = '═' * 60
        lines = ['', sep, '  FRICTION TEST 결과 요약', sep]

        if ax      is not None:
            lines.append(f'  ax_max  (직선 가속) = {ax:.2f}  m/s²'
                         f'  (μ_accel ≈ {ax/g:.3f})')
        if ab      is not None:
            lines.append(f'  a_brake (직선 제동) = {ab:.2f}  m/s²'
                         f'  (μ_brake ≈ {ab/g:.3f})')
        if ccw     is not None:
            lines.append(f'  ay_max  CCW (좌회전) = {ccw:.2f}  m/s²'
                         f'  (μ_lat ≈ {ccw/g:.3f})')
        if cw      is not None:
            lines.append(f'  ay_max   CW (우회전) = {cw:.2f}  m/s²'
                         f'  (μ_lat ≈ {cw/g:.3f})')
        if ay_mean is not None:
            lines.append(f'  ay_max  평균         = {ay_mean:.2f}  m/s²'
                         f'  (보수적: {ay_min:.2f} m/s²)')

        if ay_min is not None:
            a_tot = ay_min          # 보수적: CCW/CW 최솟값
            a_br  = ab if ab is not None else a_tot
            ax_g  = ax if ax is not None else a_tot

            lines += [
                '',
                '  ── friction_circle.yaml 업데이트 권장값 ─────────────',
                f'    a_total_max: {a_tot:.1f}',
                f'    a_brake_max: {a_br:.1f}',
                '',
                '  ── ggv.csv 전체 교체 권장값 ─────────────────────────',
                '  # v_mps, ax_max_mps2, ay_max_mps2',
                f'  0.0,  {ax_g:.2f}, {a_tot:.2f}',
                f'  3.0,  {ax_g:.2f}, {a_tot:.2f}',
                f'  6.0,  {ax_g:.2f}, {a_tot:.2f}',
                f'  9.0,  {ax_g:.2f}, {a_tot:.2f}',
                '',
                '  ggv.csv 수정 후 trajectory_optimizer 재실행 필요.',
            ]

        lines += ['', f'  CSV 저장됨 → {self._csv_path}', sep, '']
        summary = '\n'.join(lines)
        print(summary)
        self.get_logger().info(summary)

    # ── 종료 처리 ─────────────────────────────────────────────────────────────

    def destroy_node(self):
        try:                          # 비상 정지
            stop = AckermannDriveStamped()
            stop.drive.speed          = 0.0
            stop.drive.steering_angle = 0.0
            self._pub.publish(stop)
        except Exception:
            pass
        if not self._file.closed:
            self._file.flush()
            self._file.close()
        super().destroy_node()


# ── 진입점 ───────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = FrictionTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
