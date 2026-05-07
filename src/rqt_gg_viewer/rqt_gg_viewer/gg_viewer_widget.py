"""GG Viewer Widget — matplotlib + table + diff for GGV diamond results."""

import json
import copy

import numpy as np
from std_msgs.msg import String

from python_qt_binding.QtCore import Qt, Signal
from python_qt_binding.QtGui import QColor, QFont
from python_qt_binding.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QComboBox, QCheckBox, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QStatusBar, QGroupBox,
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class GGViewerWidget(QWidget):
    """Main widget: diamond plot + summary table + diff + slope tab."""

    # Thread-safe signals (ROS callback thread → Qt main thread)
    results_signal = Signal(str)
    status_signal = Signal(str)

    def __init__(self, context=None):
        super().__init__()
        self.setWindowTitle('GG Viewer')

        # ROS2: context.node 를 통해 rclpy Node 에 접근.
        # context 가 None 이면 (단독 실행 / 테스트) ROS subscribe 생략.
        self._node = getattr(context, "node", None) if context is not None else None

        self.current_data = None
        self.baseline_data = None

        self._build_ui()
        self._connect_signals()
        self._subscribe_ros()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        root = QVBoxLayout(self)

        # --- top toolbar ---
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel('V:'))
        self.v_combo = QComboBox()
        self.v_combo.setMinimumWidth(80)
        toolbar.addWidget(self.v_combo)
        toolbar.addWidget(QLabel('g:'))
        self.g_combo = QComboBox()
        self.g_combo.setMinimumWidth(80)
        toolbar.addWidget(self.g_combo)
        ## IY : slope cross-section selector for 3D slope-aware ggv
        toolbar.addWidget(QLabel('Slope:'))
        self.slope_combo = QComboBox()
        self.slope_combo.setMinimumWidth(90)
        toolbar.addWidget(self.slope_combo)
        ## IY : end
        self.diff_check = QCheckBox('Show diff')
        self.diff_check.setChecked(True)
        toolbar.addWidget(self.diff_check)
        self.baseline_btn = QPushButton('Set as Baseline')
        toolbar.addWidget(self.baseline_btn)
        self.auto_baseline_check = QCheckBox('Auto-baseline first')
        self.auto_baseline_check.setChecked(True)
        self.auto_baseline_check.setToolTip(
            'Automatically set the first received result as baseline')
        toolbar.addWidget(self.auto_baseline_check)
        toolbar.addStretch()
        root.addLayout(toolbar)

        # --- tabs ---
        self.tabs = QTabWidget()

        # Tab 1: Diamond plot + summary table
        main_tab = QWidget()
        main_layout = QHBoxLayout(main_tab)
        splitter = QSplitter(Qt.Horizontal)

        # left: matplotlib diamond plot
        plot_group = QGroupBox('GG Diamond')
        plot_layout = QVBoxLayout(plot_group)
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax_plot = self.fig.add_subplot(111)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_group)

        # right: summary table
        table_group = QGroupBox('Diamond Summary (velocity_frame)')
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget()
        self.table.setFont(QFont('Monospace', 9))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.table)
        splitter.addWidget(table_group)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        self.tabs.addTab(main_tab, 'Diamond + Table')

        # Tab 2: Slope analysis
        slope_tab = QWidget()
        slope_layout = QVBoxLayout(slope_tab)
        self.slope_fig = Figure(figsize=(6, 4), dpi=100)
        self.slope_canvas = FigureCanvas(self.slope_fig)
        self.slope_ax = self.slope_fig.add_subplot(111)
        slope_layout.addWidget(self.slope_canvas)
        self.slope_label = QLabel('No slope data yet. Enable slope in GGTuner rqt.')
        slope_layout.addWidget(self.slope_label)
        self.tabs.addTab(slope_tab, 'Slope Analysis')

        # Tab 3: Tuning params
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(3)
        self.params_table.setHorizontalHeaderLabels(['Param', 'Current', 'Baseline'])
        self.params_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        params_layout.addWidget(self.params_table)
        self.tabs.addTab(params_tab, 'Tuning Params')

        root.addWidget(self.tabs)

        # --- status bar ---
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('Waiting for /gg_results ...')
        root.addWidget(self.status_bar)

    def _connect_signals(self):
        self.results_signal.connect(self._handle_results)
        self.status_signal.connect(self._handle_status)
        self.v_combo.currentIndexChanged.connect(self._on_combo_changed)
        self.g_combo.currentIndexChanged.connect(self._on_combo_changed)
        ## IY : slope_combo selection drives diamond redraw
        self.slope_combo.currentIndexChanged.connect(self._on_combo_changed)
        ## IY : end
        self.diff_check.stateChanged.connect(self._on_combo_changed)
        self.baseline_btn.clicked.connect(self._set_baseline)

    # ------------------------------------------------------------------ #
    #  ROS subscriptions
    # ------------------------------------------------------------------ #
    def _subscribe_ros(self):
        if self._node is None:
            self._sub_results = None
            self._sub_status = None
            return
        self._sub_results = self._node.create_subscription(
            String, '/gg_results', self._on_results_cb, 1)
        self._sub_status = self._node.create_subscription(
            String, '/gg_compute_status', self._on_status_cb, 5)

    def _on_results_cb(self, msg):
        self.results_signal.emit(msg.data)

    def _on_status_cb(self, msg):
        self.status_signal.emit(msg.data)

    # ------------------------------------------------------------------ #
    #  Qt slot handlers (main thread)
    # ------------------------------------------------------------------ #
    def _handle_results(self, json_str):
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            if self._node is not None:
                self._node.get_logger().warn(f'[GGViewer] JSON parse error: {e}')
            else:
                print(f'[GGViewer] JSON parse error: {e}')
            return
        self.current_data = data

        ### IY : auto-discard incompatible baseline when v_list/g_list change
        # (e.g. v_max edited, FAST↔FULL toggled). Otherwise overlay would
        # silently compare a different (V, g) layer than the title shows.
        baseline_reset_msg = None
        if self.baseline_data is not None:
            ok, reason = self._baseline_compatible(data, self.baseline_data)
            if not ok:
                self.baseline_data = None
                baseline_reset_msg = f'Baseline reset (incompatible: {reason})'
        ### IY : end

        if self.baseline_data is None and self.auto_baseline_check.isChecked():
            self.baseline_data = copy.deepcopy(data)
        self._populate_combos(data)
        self._update_display()
        if baseline_reset_msg is not None:
            self.status_bar.showMessage(baseline_reset_msg)
        else:
            self.status_bar.showMessage(
                f"Received: {data.get('vehicle_name', '?')} "
                f"@ {data.get('timestamp', '?')}")

    def _handle_status(self, text):
        self.status_bar.showMessage(text)

    def _set_baseline(self):
        if self.current_data is not None:
            self.baseline_data = copy.deepcopy(self.current_data)
            self.status_bar.showMessage('Baseline set from current data')
            self._update_display()

    def _on_combo_changed(self, _=None):
        self._update_display()

    # ------------------------------------------------------------------ #
    #  Populate combo boxes
    # ------------------------------------------------------------------ #
    def _populate_combos(self, data):
        v_list = data.get('v_list', [])
        g_list = data.get('g_list', [])

        prev_v = self.v_combo.currentText()
        prev_g = self.g_combo.currentText()

        self.v_combo.blockSignals(True)
        self.g_combo.blockSignals(True)
        self.v_combo.clear()
        self.g_combo.clear()
        for v in v_list:
            self.v_combo.addItem(f'{v:.2f}')
        for g in g_list:
            self.g_combo.addItem(f'{g:.2f}')

        # restore previous selection if possible
        idx_v = self.v_combo.findText(prev_v)
        if idx_v >= 0:
            self.v_combo.setCurrentIndex(idx_v)
        idx_g = self.g_combo.findText(prev_g)
        if idx_g >= 0:
            self.g_combo.setCurrentIndex(idx_g)

        self.v_combo.blockSignals(False)
        self.g_combo.blockSignals(False)

        ## IY : populate slope_combo from slope_results.slope_list_deg
        slope_data = data.get('slope_results')
        prev_slope = self.slope_combo.currentText()
        self.slope_combo.blockSignals(True)
        self.slope_combo.clear()
        if (slope_data is not None
                and isinstance(slope_data, dict)
                and 'slope_list_deg' in slope_data
                and len(slope_data.get('slope_list_deg', [])) > 0):
            s_list = slope_data['slope_list_deg']
            labels = [f'{float(s):+.1f}°' for s in s_list]
            for lbl in labels:
                self.slope_combo.addItem(lbl)
            if prev_slope and prev_slope in labels:
                self.slope_combo.setCurrentIndex(labels.index(prev_slope))
            else:
                # default to slope=0 (closest)
                s_arr = np.asarray(s_list, dtype=float)
                self.slope_combo.setCurrentIndex(int(np.argmin(np.abs(s_arr))))
            self.slope_combo.setEnabled(True)
        else:
            self.slope_combo.addItem('flat (no sweep)')
            self.slope_combo.setEnabled(False)
        self.slope_combo.blockSignals(False)
        ## IY : end

    # ------------------------------------------------------------------ #
    #  Update all displays
    # ------------------------------------------------------------------ #
    def _update_display(self):
        if self.current_data is None:
            return
        self._update_table()
        self._update_plot()
        self._update_slope_tab()
        self._update_params_tab()

    # ------------------------------------------------------------------ #
    #  Summary table
    # ------------------------------------------------------------------ #
    def _update_table(self):
        d = self.current_data['diamond']
        v_list = self.current_data['v_list']
        g_list = self.current_data['g_list']
        ### IY : same compatibility gate as _update_plot — only show diff cells
        # when v_list/g_list match (otherwise [vi][gi] indexes a different
        # (V, g) layer in baseline than in current and the diff is misleading)
        compat_ok, _ = self._baseline_compatible(
            self.current_data, self.baseline_data)
        show_diff = (self.diff_check.isChecked()
                     and self.baseline_data is not None
                     and compat_ok)
        ### IY : end

        n_v = len(v_list)
        n_g = len(g_list)
        # rows = velocities, columns = g values × 3 metrics
        metrics = ['ax_max', 'ax_min', 'ay_max']
        n_cols = n_g * len(metrics)

        self.table.setRowCount(n_v)
        self.table.setColumnCount(n_cols)

        # headers
        col_headers = []
        for g in g_list:
            for m in metrics:
                col_headers.append(f'g={g:.1f}\n{m}')
        self.table.setHorizontalHeaderLabels(col_headers)

        row_headers = [f'V={v:.1f}' for v in v_list]
        self.table.setVerticalHeaderLabels(row_headers)

        base_d = self.baseline_data['diamond'] if show_diff else None

        for vi in range(n_v):
            for gi in range(n_g):
                for mi, m_key in enumerate(metrics):
                    col = gi * len(metrics) + mi
                    val = d[m_key][vi][gi]
                    text = f'{val:.3f}'

                    item = QTableWidgetItem()
                    if show_diff and base_d is not None:
                        base_val = self._get_baseline_val(base_d, m_key, vi, gi)
                        if base_val is not None:
                            delta = val - base_val
                            pct = (delta / abs(base_val) * 100) if abs(base_val) > 1e-9 else 0
                            text = f'{val:.3f}\n({delta:+.3f} {pct:+.1f}%)'
                            if delta > 0.001:
                                item.setBackground(QColor(200, 255, 200))  # green
                            elif delta < -0.001:
                                item.setBackground(QColor(255, 200, 200))  # red
                    item.setText(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(vi, col, item)

    def _get_baseline_val(self, base_d, key, vi, gi):
        try:
            return base_d[key][vi][gi]
        except (IndexError, KeyError):
            return None

    ### IY : baseline compatibility check
    # Two snapshots are comparable index-wise only if their v_list/g_list
    # match exactly (length AND values). When they differ (v_max edited,
    # FAST↔FULL toggled), [vi][gi] points at a different (V, g) layer in
    # baseline than in current — the overlay would silently lie. Disable
    # comparison instead of papering over with nearest-neighbor mapping.
    @staticmethod
    def _baseline_compatible(current, baseline, atol=1e-6):
        if baseline is None:
            return False, 'no baseline loaded'
        cv = np.asarray(current.get('v_list', []), dtype=float)
        bv = np.asarray(baseline.get('v_list', []), dtype=float)
        cg = np.asarray(current.get('g_list', []), dtype=float)
        bg = np.asarray(baseline.get('g_list', []), dtype=float)
        if cv.shape != bv.shape:
            return False, f'v_list length differs ({len(cv)} vs {len(bv)})'
        if cg.shape != bg.shape:
            return False, f'g_list length differs ({len(cg)} vs {len(bg)})'
        if cv.size and not np.allclose(cv, bv, atol=atol):
            return False, (f'v_list values differ '
                           f'(cur=[{cv[0]:.2f}…{cv[-1]:.2f}], '
                           f'base=[{bv[0]:.2f}…{bv[-1]:.2f}])')
        if cg.size and not np.allclose(cg, bg, atol=atol):
            return False, (f'g_list values differ '
                           f'(cur=[{cg[0]:.1f}…{cg[-1]:.1f}], '
                           f'base=[{bg[0]:.1f}…{bg[-1]:.1f}])')
        return True, ''
    ### IY : end

    # ------------------------------------------------------------------ #
    #  Diamond GG plot
    # ------------------------------------------------------------------ #
    def _update_plot(self):
        self.ax_plot.clear()
        d = self.current_data['diamond']
        v_list = self.current_data['v_list']
        g_list = self.current_data['g_list']

        vi = self.v_combo.currentIndex()
        gi = self.g_combo.currentIndex()
        if vi < 0 or gi < 0:
            self.canvas.draw()
            return

        v_val = v_list[vi]
        g_val = g_list[gi]

        ## IY : 3D slope-aware diamond if slope_results carries gg_exponent
        slope_data = self.current_data.get('slope_results')
        si = self.slope_combo.currentIndex()
        use_3d = (slope_data is not None
                  and isinstance(slope_data, dict)
                  and 'slope_list_deg' in slope_data
                  and 'gg_exponent' in slope_data
                  and self.slope_combo.isEnabled()
                  and si >= 0)
        title_suffix = ''
        if use_3d:
            try:
                ax_max_val = slope_data['ax_max'][vi][gi][si]
                ax_min_val = slope_data['ax_min'][vi][gi][si]
                ay_max_val = slope_data['ay_max'][vi][gi][si]
                gg_exp     = slope_data['gg_exponent'][vi][gi][si]
                slope_deg  = float(slope_data['slope_list_deg'][si])
                title_suffix = f'  slope={slope_deg:+.1f}°'
            except (IndexError, KeyError, TypeError):
                ax_max_val = d['ax_max'][vi][gi]
                ax_min_val = d['ax_min'][vi][gi]
                ay_max_val = d['ay_max'][vi][gi]
                gg_exp     = d['gg_exponent'][vi][gi]
                title_suffix = '  (3d→flat fallback)'
                use_3d = False
        else:
            ax_max_val = d['ax_max'][vi][gi]
            ax_min_val = d['ax_min'][vi][gi]
            ay_max_val = d['ay_max'][vi][gi]
            gg_exp     = d['gg_exponent'][vi][gi]
        ## IY : end

        # Draw current diamond
        self._draw_diamond(self.ax_plot, ax_max_val, ax_min_val, ay_max_val,
                           gg_exp, color='#2196F3', linestyle='-',
                           label='Current', alpha=1.0)

        # Draw baseline diamond
        ### IY : require v_list/g_list compatibility (defensive — _handle_results
        # already discards incompatible baseline, but a stale caller could ask)
        compat_ok, _ = self._baseline_compatible(
            self.current_data, self.baseline_data)
        show_diff = (self.diff_check.isChecked()
                     and self.baseline_data is not None
                     and compat_ok)
        ### IY : end
        if show_diff:
            bd = self.baseline_data['diamond']
            ## IY : baseline matches slope index when both have 3D data
            b_slope_data = self.baseline_data.get('slope_results')
            b_use_3d = (use_3d
                        and b_slope_data is not None
                        and isinstance(b_slope_data, dict)
                        and 'gg_exponent' in b_slope_data)
            try:
                if b_use_3d:
                    b_ax_max = b_slope_data['ax_max'][vi][gi][si]
                    b_ax_min = b_slope_data['ax_min'][vi][gi][si]
                    b_ay_max = b_slope_data['ay_max'][vi][gi][si]
                    b_gg_exp = b_slope_data['gg_exponent'][vi][gi][si]
                else:
                    b_ax_max = bd['ax_max'][vi][gi]
                    b_ax_min = bd['ax_min'][vi][gi]
                    b_ay_max = bd['ay_max'][vi][gi]
                    b_gg_exp = bd['gg_exponent'][vi][gi]
                ## IY : end
                self._draw_diamond(self.ax_plot, b_ax_max, b_ax_min, b_ay_max,
                                   b_gg_exp, color='gray', linestyle='--',
                                   label='Baseline', alpha=0.7)
            except (IndexError, KeyError, TypeError):
                pass

        self.ax_plot.set_xlabel('Lateral ay [m/s²]')
        self.ax_plot.set_ylabel('Longitudinal ax [m/s²]')
        self.ax_plot.set_title(
            f'GG Diamond  V={v_val:.1f} m/s  g={g_val:.1f} m/s²{title_suffix}')
        self.ax_plot.legend(loc='upper right', fontsize=8)
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.set_aspect('equal', adjustable='datalim')
        self.ax_plot.axhline(y=0, color='k', linewidth=0.5)
        self.ax_plot.axvline(x=0, color='k', linewidth=0.5)
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_diamond(self, ax, ax_max, ax_min, ay_max, gg_exp,
                      color='blue', linestyle='-', label=None, alpha=1.0):
        """Draw a GG diamond envelope using the diamond parameterization."""
        exp = max(gg_exp, 0.5)
        theta = np.linspace(0, 2 * np.pi, 200)
        # diamond parameterization: |ax/ax_lim|^exp + |ay/ay_max|^exp = 1
        ay_pts = ay_max * np.cos(theta)
        ax_pts = np.zeros_like(theta)
        for i, t in enumerate(theta):
            ay_frac = min(abs(ay_pts[i]) / ay_max, 1.0) if ay_max > 1e-9 else 0
            remaining = max(1.0 - ay_frac ** exp, 0.0)
            ax_lim = ax_max if np.sin(t) >= 0 else abs(ax_min)
            ax_pts[i] = ax_lim * remaining ** (1.0 / exp) * np.sign(np.sin(t))
        ax.plot(ay_pts, ax_pts, color=color, linestyle=linestyle,
                linewidth=2, label=label, alpha=alpha)

    # ------------------------------------------------------------------ #
    #  Slope tab
    # ------------------------------------------------------------------ #
    def _update_slope_tab(self):
        slope_data = self.current_data.get('slope_results')
        if slope_data is None:
            self.slope_label.setText(
                'No slope data. Enable "enable_slope" in GGTuner rqt.')
            self.slope_label.show()
            self.slope_ax.clear()
            self.slope_canvas.draw()
            return
        self.slope_label.hide()
        self.slope_ax.clear()

        vi = self.v_combo.currentIndex()
        gi = self.g_combo.currentIndex()
        if vi < 0 or gi < 0:
            self.slope_canvas.draw()
            return

        slopes_deg = slope_data.get('slope_list_deg', [])
        ax_max_s = slope_data.get('ax_max', [])
        ax_min_s = slope_data.get('ax_min', [])
        ay_max_s = slope_data.get('ay_max', [])

        try:
            ax_max_vs = [ax_max_s[vi][gi][si] for si in range(len(slopes_deg))]
            ax_min_vs = [ax_min_s[vi][gi][si] for si in range(len(slopes_deg))]
            ay_max_vs = [ay_max_s[vi][gi][si] for si in range(len(slopes_deg))]
        except (IndexError, KeyError):
            self.slope_label.setText('Slope data dimension mismatch.')
            self.slope_label.show()
            self.slope_canvas.draw()
            return

        v_val = self.current_data['v_list'][vi]
        g_val = self.current_data['g_list'][gi]

        self.slope_ax.plot(slopes_deg, ax_max_vs, 'r-o', markersize=4, label='ax_max')
        self.slope_ax.plot(slopes_deg, ax_min_vs, 'b-o', markersize=4, label='ax_min')
        self.slope_ax.plot(slopes_deg, ay_max_vs, 'g-o', markersize=4, label='ay_max')
        self.slope_ax.set_xlabel('Slope angle [deg]')
        self.slope_ax.set_ylabel('Acceleration [m/s²]')
        self.slope_ax.set_title(f'Slope effect  V={v_val:.1f} m/s  g={g_val:.1f} m/s²')
        self.slope_ax.legend(fontsize=8)
        self.slope_ax.grid(True, alpha=0.3)
        self.slope_ax.axvline(x=0, color='k', linewidth=0.5, linestyle='--')
        self.slope_fig.tight_layout()
        self.slope_canvas.draw()

    # ------------------------------------------------------------------ #
    #  Tuning params tab
    # ------------------------------------------------------------------ #
    def _update_params_tab(self):
        cur_p = self.current_data.get('tuning_params', {})
        base_p = self.baseline_data.get('tuning_params', {}) if self.baseline_data else {}
        all_keys = sorted(set(list(cur_p.keys()) + list(base_p.keys())))

        self.params_table.setRowCount(len(all_keys))
        for i, key in enumerate(all_keys):
            self.params_table.setItem(i, 0, QTableWidgetItem(key))

            cur_val = cur_p.get(key, '')
            base_val = base_p.get(key, '')
            cur_item = QTableWidgetItem(f'{cur_val}')
            base_item = QTableWidgetItem(f'{base_val}')

            if cur_val != '' and base_val != '' and cur_val != base_val:
                cur_item.setBackground(QColor(255, 255, 180))  # highlight changed
            self.params_table.setItem(i, 1, cur_item)
            self.params_table.setItem(i, 2, base_item)

    # ------------------------------------------------------------------ #
    #  Shutdown
    # ------------------------------------------------------------------ #
    def shutdown(self):
        if self._node is None:
            return
        if self._sub_results is not None:
            self._node.destroy_subscription(self._sub_results)
        if self._sub_status is not None:
            self._node.destroy_subscription(self._sub_status)
