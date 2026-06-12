#!/usr/bin/env python3

import os
import sys
import copy
import json
import shutil
import struct
import subprocess
import threading
import time
from datetime import datetime

import numpy as np
import yaml
from std_msgs.msg import String
## IY : Trigger service for FBGA hot-reload
from std_srvs.srv import Trigger
## IY : end


class GGTunerNode:

    TIRE_KEYS = ['lambda_mu_x', 'lambda_mu_y', 'p_Dx_1', 'p_Dy_1', 'p_Dx_2', 'p_Dy_2']
    VEHICLE_KEYS = ['P_max', 'v_max', 'epsilon',
                    'P_brake_max', 'ax_max_cap', 'ax_min_cap', 'ay_max_cap']
    CAP_KEYS = ['P_brake_max', 'ax_max_cap', 'ax_min_cap', 'ay_max_cap']
    POST_KEYS = ['gg_exp_scale', 'ax_max_scale', 'ax_min_scale', 'ay_scale']
    RACELINE_KEYS = ['V_min', 'safety_distance', 'w_T', 'w_jx', 'w_jy', 'w_dOmega_z']
    ALL_TUNING_KEYS = TIRE_KEYS + VEHICLE_KEYS + POST_KEYS + RACELINE_KEYS
    ## IY : end

    def __init__(self):
        self.get_logger().info("[GGTuner] Initializing...")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        race_stack_root = os.path.dirname(os.path.dirname(script_dir))
        self.race_stack_root = race_stack_root

        self.data_path = os.path.join(
            race_stack_root, 'planner', '3d_gb_optimizer', 'global_line', 'data')
        # Canonical map storage: stack_master/maps/ (same package).
        # realpath() resolves symlink-install back to source.
        self.maps_dir = os.path.realpath(
            os.path.join(script_dir, '..', 'maps'))

        self.fast_ggv_dir = os.path.join(
            race_stack_root, 'planner', '3d_gb_optimizer', 'fast_ggv_gen')
        self.fast_ggv_script = os.path.join(self.fast_ggv_dir, 'run_on_container.sh')
        self.fast_ggv_output_dir = os.path.join(self.fast_ggv_dir, 'output')

        self.base_vehicle = self._get_param_or_default("~base_vehicle", "rc_car_10th")

        base_yml_path = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + self.base_vehicle + '.yml')
        backup_yml_path = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + self.base_vehicle + '_backup.yml')

        loaded_path = None
        for path in [base_yml_path, backup_yml_path]:
            if os.path.exists(path) and not os.path.islink(path) or \
               (os.path.islink(path) and os.path.exists(os.readlink(path) if not os.path.isabs(os.readlink(path)) else path)):
                try:
                    with open(path, 'r') as f:
                        self.base_params = yaml.safe_load(f)
                    loaded_path = path
                    break
                except (yaml.YAMLError, OSError):
                    continue
        if loaded_path is None:
            self.get_logger().error(f"[GGTuner] Base params not found: tried {base_yml_path} and {backup_yml_path}")
            raise FileNotFoundError(base_yml_path)
        self.get_logger().info(f"[GGTuner] Base params loaded: {loaded_path}")
        ## IY : end

        ## IY : scan available maps
        self.available_maps = self._scan_maps_dir()
        self.get_logger().info(f"[GGTuner] Available maps ({len(self.available_maps)}): "
                      f"{self.available_maps}")

        if not os.path.exists(self.fast_ggv_script):
            self.get_logger().warning(f"[GGTuner] fast_ggv script missing: {self.fast_ggv_script}")
        ## IY : end

        self.status_pub = rospy.Publisher(
            '/gg_compute_status', String, queue_size=5, latch=True)
        self.status_pub.publish(f"READY: {self.base_vehicle}")

        ## IY : publish GGV diamond results as JSON for rqt_gg_viewer
        self.results_pub = rospy.Publisher(
            '/gg_results', String, queue_size=1, latch=True)
        ## IY : end

        self.pipeline_thread = None
        self.pipeline_lock = threading.Lock()
        self.fbga_proc = None

        rospy.on_shutdown(self._shutdown_cleanup)

        self.srv = Server(GGTunerConfig, self.reconfigure_cb)
        self.get_logger().info("[GGTuner] Ready. Use rqt_reconfigure → /gg_tuner")

    def _round_tuning(self, tuning_dict):
        return {k: round(v, 4) for k, v in sorted(tuning_dict.items())}

    def _find_cached(self, tuning_dict):
        rounded = self._round_tuning(tuning_dict)
        gg_dir = os.path.join(self.data_path, 'gg_diagrams')
        prefix = self.base_vehicle + '_v'
        if not os.path.exists(gg_dir):
            return None
        for name in sorted(os.listdir(gg_dir)):
            if not name.startswith(prefix):
                continue
            full = os.path.join(gg_dir, name)
            if os.path.islink(full):
                continue
            suffix = name[len(prefix):]
            if not suffix.isdigit():
                continue
            meta_path = os.path.join(full, 'params_used.json')
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                saved = {k: round(v, 4) for k, v in sorted(meta['tuning'].items())}
                if saved == rounded:
                    return name
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    def _next_version(self):
        gg_dir = os.path.join(self.data_path, 'gg_diagrams')
        prefix = self.base_vehicle + '_v'
        max_ver = 0
        if os.path.exists(gg_dir):
            for name in os.listdir(gg_dir):
                if not name.startswith(prefix):
                    continue
                suffix = name[len(prefix):]
                if not suffix.isdigit():
                    continue
                max_ver = max(max_ver, int(suffix))
        return max_ver + 1
    
    def _merge_all_params(self, tuning_dict):
        merged = copy.deepcopy(self.base_params)
        # NLP: tire
        for key in self.TIRE_KEYS:
            if key in tuning_dict:
                merged['tire_params'][key] = tuning_dict[key]
        # NLP: vehicle (caps: 0.0 → None)
        for key in self.VEHICLE_KEYS:
            if key not in tuning_dict:
                continue
            val = tuning_dict[key]
            if key in self.CAP_KEYS and float(val) <= 0.0:
                merged['vehicle_params'][key] = None
            else:
                merged['vehicle_params'][key] = val
        # Post-process + raceline → top-level
        for key in self.POST_KEYS + self.RACELINE_KEYS:
            if key in tuning_dict:
                merged[key] = float(tuning_dict[key])
        return merged

    def _save_params_yml(self, vehicle_name, merged_params):
        yml_path = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + vehicle_name + '.yml')
        with open(yml_path, 'w') as f:
            yaml.dump(merged_params, f, default_flow_style=False, allow_unicode=True)
        ## IY : copy to latest (real file, not symlink)
        latest_path = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + self.base_vehicle + '_latest.yml')
        shutil.copy2(yml_path, latest_path)
        ## IY : end
        self.get_logger().info(f"[GGTuner] Params saved: {yml_path} → latest copied")
        return yml_path

    def _save_meta(self, vehicle_name, tuning_dict):
        out_dir = os.path.join(self.data_path, 'gg_diagrams', vehicle_name)
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            'vehicle_name': vehicle_name,
            'base_vehicle': self.base_vehicle,
            'tuning': tuning_dict,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        meta_path = os.path.join(out_dir, 'params_used.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f"[GGTuner] Meta saved: {meta_path}")

    def _write_latest_params_yml(self, merged_params):
        latest_yml = os.path.join(
            self.data_path, 'vehicle_params',
            f'params_{self.base_vehicle}_latest.yml')
        with open(latest_yml, 'w') as f:
            yaml.dump(merged_params, f, default_flow_style=False,
                      allow_unicode=True)
        self.get_logger().info(f"[GGTuner] latest yml written: {latest_yml}")
        return latest_yml

    ## IY : overlay rqt RACELINE_KEYS onto existing latest yml in-place
    def _update_raceline_keys_in_yml(self, vehicle_name, tuning):
        yml_path = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + vehicle_name + '.yml')
        if not os.path.exists(yml_path):
            self.get_logger().error(f"[GGTuner] raceline-keys overlay: yml missing: {yml_path}")
            return False
        try:
            with open(yml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as e:
            self.get_logger().error(f"[GGTuner] raceline-keys overlay: yml load failed: {e}")
            return False
        overlaid = {}
        for k in self.RACELINE_KEYS:
            if k in tuning:
                data[k] = float(tuning[k])
                overlaid[k] = data[k]
        try:
            with open(yml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except OSError as e:
            self.get_logger().error(f"[GGTuner] raceline-keys overlay: yml write failed: {e}")
            return False
        self.get_logger().info(f"[GGTuner] raceline keys merged into {os.path.basename(yml_path)}: {overlaid}")
        return True
    ## IY : end

    ## IY : copy snapshot GGV+yml into _latest slot (snapshot remains unchanged)
    def _activate_snapshot_as_latest(self, snapshot_name):
        latest_name = f'{self.base_vehicle}_latest'
        src_gg = os.path.join(self.data_path, 'gg_diagrams', snapshot_name)
        dst_gg = os.path.join(self.data_path, 'gg_diagrams', latest_name)
        src_yml = os.path.join(self.data_path, 'vehicle_params',
                               f'params_{snapshot_name}.yml')
        dst_yml = os.path.join(self.data_path, 'vehicle_params',
                               f'params_{latest_name}.yml')
        if not os.path.isdir(src_gg):
            self.get_logger().error(f"[GGTuner] snapshot gg_diagrams missing: {src_gg}")
            return False
        if not os.path.exists(src_yml):
            self.get_logger().error(f"[GGTuner] snapshot yml missing: {src_yml}")
            return False
        self._replace_dir(dst_gg, src_gg)
        shutil.copy2(src_yml, dst_yml)
        self.get_logger().info(f"[GGTuner] activated snapshot '{snapshot_name}' as _latest")
        self.status_pub.publish(f"RACELINE_GGV_ACTIVATED: {snapshot_name}")
        return True
    ## IY : end

    def _restore_to_latest(self, cached_name):
        latest_name = f'{self.base_vehicle}_latest'
        src_gg = os.path.join(self.data_path, 'gg_diagrams', cached_name)
        dst_gg = os.path.join(self.data_path, 'gg_diagrams', latest_name)
        src_yml = os.path.join(self.data_path, 'vehicle_params',
                               f'params_{cached_name}.yml')
        dst_yml = os.path.join(self.data_path, 'vehicle_params',
                               f'params_{latest_name}.yml')
        if not os.path.exists(src_gg):
            self.get_logger().error(f"[GGTuner] cached gg_diagrams missing: {src_gg}")
            return False
        if os.path.islink(dst_gg):
            os.unlink(dst_gg)
        elif os.path.exists(dst_gg):
            shutil.rmtree(dst_gg)
        shutil.copytree(src_gg, dst_gg)
        if os.path.exists(src_yml):
            shutil.copy2(src_yml, dst_yml)
        self.get_logger().info(f"[GGTuner] restored {cached_name} → latest")
        return True

    def _snapshot_latest_to_version(self, tuning_dict=None):
        ver = self._next_version()
        snapshot_name = f'{self.base_vehicle}_v{ver}'
        self.get_logger().info(f"[GGTuner] ===== SAVE snapshot: {snapshot_name} =====")
        self.status_pub.publish(f"SAVING: {snapshot_name}")

        latest_name = f'{self.base_vehicle}_latest'
        latest_gg = os.path.join(self.data_path, 'gg_diagrams', latest_name)
        latest_yml = os.path.join(self.data_path, 'vehicle_params',
                                  f'params_{latest_name}.yml')
        if not os.path.exists(latest_gg):
            self.get_logger().error(f"[GGTuner] SAVE failed: {latest_gg} missing")
            self.status_pub.publish("SAVE_FAILED: no latest gg")
            return False
        if not os.path.exists(latest_yml):
            self.get_logger().error(f"[GGTuner] SAVE failed: {latest_yml} missing")
            self.status_pub.publish("SAVE_FAILED: no latest yml")
            return False

        dst_gg = os.path.join(self.data_path, 'gg_diagrams', snapshot_name)
        real_src = (os.path.realpath(latest_gg)
                    if os.path.islink(latest_gg) else latest_gg)
        shutil.copytree(real_src, dst_gg)
        dst_yml = os.path.join(self.data_path, 'vehicle_params',
                               f'params_{snapshot_name}.yml')
        shutil.copy2(latest_yml, dst_yml)

        latest_meta_path = os.path.join(latest_gg, 'params_used.json')
        meta = None
        if os.path.exists(latest_meta_path):
            try:
                with open(latest_meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                meta = None
        if meta is None:
            meta = {
                'vehicle_name': snapshot_name,
                'base_vehicle': self.base_vehicle,
                'tuning': tuning_dict or {},
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
        meta['vehicle_name'] = snapshot_name
        meta['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        meta_path = os.path.join(dst_gg, 'params_used.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self.get_logger().info(f"[GGTuner] SAVED: {snapshot_name}")
        self.status_pub.publish(f"SAVED: {snapshot_name}")
        return True
    ## IY : end

    def _scan_maps_dir(self):
        if not os.path.exists(self.maps_dir):
            return []
        return sorted([n for n in os.listdir(self.maps_dir)
                       if os.path.isdir(os.path.join(self.maps_dir, n))])

    def _generate_gg_bin(self, npy_dir):
        bin_path = os.path.join(npy_dir, 'gg.bin')
        try:
            v_list = np.load(os.path.join(npy_dir, 'v_list.npy')).astype(np.float64)
            g_list = np.load(os.path.join(npy_dir, 'g_list.npy')).astype(np.float64)
            ax_max = np.load(os.path.join(npy_dir, 'ax_max.npy')).astype(np.float64)
            ax_min = np.load(os.path.join(npy_dir, 'ax_min.npy')).astype(np.float64)
            ay_max = np.load(os.path.join(npy_dir, 'ay_max.npy')).astype(np.float64)
            gg_exp = np.load(os.path.join(npy_dir, 'gg_exponent.npy')).astype(np.float64)
        except (OSError, FileNotFoundError) as e:
            self.get_logger().error(f"[GGTuner] gg.bin gen failed (npy missing): {e}")
            return False

        ## IY : detect 3D slope data and write 3D gg.bin if available
        slope_list_path = os.path.join(npy_dir, 'slope_list.npy')
        ax_max_3d_path = os.path.join(npy_dir, 'ax_max_3d.npy')
        has_3d = os.path.exists(slope_list_path) and os.path.exists(ax_max_3d_path)

        if has_3d:
            slope_list = np.load(slope_list_path).astype(np.float64)
            ax_max_3d = np.load(ax_max_3d_path).astype(np.float64)
            ax_min_3d = np.load(os.path.join(npy_dir, 'ax_min_3d.npy')).astype(np.float64)
            ay_max_3d = np.load(os.path.join(npy_dir, 'ay_max_3d.npy')).astype(np.float64)
            gg_exp_3d = np.load(os.path.join(npy_dir, 'gg_exponent_3d.npy')).astype(np.float64)
            nv, ng, ns = len(v_list), len(g_list), len(slope_list)
            try:
                with open(bin_path, 'wb') as f:
                    f.write(struct.pack('III', nv, ng, ns))
                    for arr in [v_list, g_list, slope_list]:
                        arr.tofile(f)
                    # 3D arrays: row-major [nv, ng, ns]
                    for arr in [ax_max_3d, ax_min_3d, ay_max_3d, gg_exp_3d]:
                        arr.astype(np.float64).tofile(f)
            except OSError as e:
                self.get_logger().error(f"[GGTuner] 3D gg.bin write failed: {e}")
                return False
            self.get_logger().info(
                f"[GGTuner] 3D gg.bin written: {bin_path} "
                f"(nv={nv}, ng={ng}, ns={ns}, "
                f"size={os.path.getsize(bin_path)} B)")
        else:
            nv, ng = len(v_list), len(g_list)
            try:
                with open(bin_path, 'wb') as f:
                    f.write(struct.pack('II', nv, ng))
                    for arr in [v_list, g_list, ax_max, ax_min, ay_max, gg_exp]:
                        arr.tofile(f)
            except OSError as e:
                self.get_logger().error(f"[GGTuner] gg.bin write failed: {e}")
                return False
            self.get_logger().info(
                f"[GGTuner] 2D gg.bin written: {bin_path} (nv={nv}, ng={ng}, "
                f"size={os.path.getsize(bin_path)} B)")
        ## IY : end
        return True

    ## IY : save ggv meta (slope_ax_scale, enable_slope) for downstream auto-pairing
    ##      consumers: fast_sqp_planner._load_ggv, global_velocity_planner_3d
    ##      also cleans stale *_3d.npy when enable_slope=False
    def _save_ggv_meta(self, vehicle_name, slope_ax_scale, enable_slope):
        import numpy as np
        dst = os.path.join(self.data_path, 'gg_diagrams', vehicle_name)
        for frame in ('velocity_frame', 'vehicle_frame'):
            frame_dir = os.path.join(dst, frame)
            if not os.path.isdir(frame_dir):
                continue
            try:
                np.save(os.path.join(frame_dir, 'slope_ax_scale.npy'),
                        np.float64(slope_ax_scale))
                np.save(os.path.join(frame_dir, 'enable_slope.npy'),
                        np.bool_(enable_slope))
            except Exception as e:
                self.get_logger().warning(f"[GGTuner] save ggv meta failed ({frame}): {e}")
            ## IY : cleanup stale *_3d.npy when enable_slope=False
            ##      slope_data/ rho cache is preserved for future enable_slope=True reuse
            if not enable_slope:
                stale = ('ax_max_3d.npy', 'ax_min_3d.npy', 'ay_max_3d.npy',
                         'gg_exponent_3d.npy', 'slope_list.npy',
                         'slope_list_deg.npy')
                removed = []
                for fn in stale:
                    p = os.path.join(frame_dir, fn)
                    if os.path.isfile(p):
                        try:
                            os.remove(p)
                            removed.append(fn)
                        except OSError as e:
                            self.get_logger().warning(f"[GGTuner] cleanup failed ({fn}): {e}")
                if removed:
                    self.get_logger().info(
                        f"[GGTuner] cleaned stale 3D npy in {frame}: {removed}")
            ## IY : end
        self.get_logger().info(f"[GGTuner] ggv meta saved: slope_ax_scale={slope_ax_scale}, enable_slope={enable_slope}")
    ## IY : end

    def _copy_to_gg_diagrams(self, vehicle_name):
        src = os.path.join(self.fast_ggv_output_dir, vehicle_name)
        dst = os.path.join(self.data_path, 'gg_diagrams', vehicle_name)
        if not os.path.exists(src):
            self.get_logger().error(f"[GGTuner] fast_ggv output missing: {src}")
            return False
        meta_backup = None
        meta_path = os.path.join(dst, 'params_used.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta_backup = f.read()
            except OSError:
                pass
        if os.path.exists(dst) or os.path.islink(dst):
            if os.path.islink(dst):
                os.unlink(dst)
            else:
                shutil.rmtree(dst)
        shutil.copytree(src, dst)
        if meta_backup is not None:
            with open(meta_path, 'w') as f:
                f.write(meta_backup)
        self.get_logger().info(f"[GGTuner] Copied: {vehicle_name} → gg_diagrams/")
        for frame in ('velocity_frame', 'vehicle_frame'):
            frame_dir = os.path.join(dst, frame)
            if os.path.isdir(frame_dir):
                self._generate_gg_bin(frame_dir)
        return True

    ## IY : symlinks for DIRECTORIES only (params latest is a real file copy)
    def _update_dir_symlinks(self, vehicle_name):
        latest_name = f'{self.base_vehicle}_latest'
        targets = [
            (self.fast_ggv_output_dir, latest_name, vehicle_name),
            (os.path.join(self.data_path, 'gg_diagrams'), latest_name, vehicle_name),
        ]
        for parent, link_name, target in targets:
            link_path = os.path.join(parent, link_name)
            target_path = os.path.join(parent, target)
            if not os.path.exists(target_path):
                continue
            try:
                if os.path.islink(link_path) or os.path.exists(link_path):
                    os.unlink(link_path)
                os.symlink(target, link_path)
                self.get_logger().info(f"[GGTuner] symlink: {link_name} → {target}")
            except OSError as e:
                self.get_logger().warning(f"[GGTuner] symlink failed: {e}")
    ## IY : end

    def _run_and_stream(self, cmd, tag, timeout=600, env=None):
        self.get_logger().info(f"[GGTuner] [{tag}] cmd: {' '.join(cmd)}")
        run_env = None
        if env is not None:
            run_env = os.environ.copy()
            run_env.update(env)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=run_env)
        except (FileNotFoundError, OSError) as e:
            self.get_logger().error(f"[GGTuner] [{tag}] Popen failed: {e}")
            return False
        start_time = time.time()
        try:
            for line in iter(proc.stdout.readline, ''):
                if time.time() - start_time > timeout:
                    self.get_logger().error(f"[GGTuner] [{tag}] timeout ({timeout}s)")
                    proc.kill()
                    return False
                line = line.rstrip()
                if line:
                    self.get_logger().info(f"[{tag}] {line}")
        except Exception as e:
            self.get_logger().error(f"[GGTuner] [{tag}] stream error: {e}")
            proc.kill()
            return False
        proc.wait()
        ok = (proc.returncode == 0)
        if ok:
            self.get_logger().info(f"[GGTuner] [{tag}] done (rc=0)")
        else:
            self.get_logger().error(f"[GGTuner] [{tag}] failed (rc={proc.returncode})")
        return ok

    ## IY : fast_ggv — no --tuning, unified params yml already has everything
    def _run_fast_ggv(self, vehicle_name, full_resolution=False,
                      enable_slope=False, slope_max_deg=15.0, slope_N=5,
                      slope_ax_scale=1.0, slope_normal_scale=1.0):
        self.get_logger().info(f"[GGTuner] [fast_ggv] starting: {vehicle_name}")
        self.status_pub.publish(f"GGV_COMPUTING: {vehicle_name}")
        if not os.path.exists(self.fast_ggv_script):
            self.get_logger().error(f"[GGTuner] fast_ggv script missing: {self.fast_ggv_script}")
            return False
        resolution = '--full' if full_resolution else '--fast'
        cmd = ['bash', self.fast_ggv_script, vehicle_name, resolution]
        env = {}
        if enable_slope:
            env['ENABLE_SLOPE'] = '1'
            env['SLOPE_MAX_DEG'] = str(slope_max_deg)
            env['SLOPE_N'] = str(slope_N)
            env['SLOPE_AX_SCALE'] = str(slope_ax_scale)
            env['SLOPE_NORMAL_SCALE'] = str(slope_normal_scale)
        return self._run_and_stream(cmd, tag='fast_ggv', timeout=600,
                                    env=env if env else None)
    ## IY : end

    def _read_friction_sectors(self, map_name):
        """Read friction sectors — rosparam first (live rqt values), yaml fallback."""
        # 1) try rosparam (set by friction_sector_server rqt)
        try:
            n_sec = self._get_param_or_default('/friction_map_params/n_sectors', 0)
            if n_sec > 0:
                sectors = []
                for i in range(n_sec):
                    sectors.append({
                        'start': int(self._get_param_or_default(f'/friction_map_params/Sector{i}/start', 0)),
                        'end':   int(self._get_param_or_default(f'/friction_map_params/Sector{i}/end', 0)),
                        'friction': float(self._get_param_or_default(f'/friction_map_params/Sector{i}/friction', -1.0)),
                    })
                valid = [s for s in sectors if s['friction'] > 0]
                if valid:
                    self.get_logger().info(f"[GGTuner] Friction sectors from rosparam: {len(valid)} sectors")
                    return valid
        except Exception:
            pass
        # 2) fallback: read yaml file
        yaml_path = os.path.join(self.maps_dir, map_name, 'friction_scaling.yaml')
        if not os.path.exists(yaml_path):
            self.get_logger().info(f"[GGTuner] No friction sectors for {map_name}")
            return []
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            sectors = []
            for i in range(data.get('n_sectors', 0)):
                sec = data.get(f'Sector{i}', {})
                fric = sec.get('friction', -1.0)
                if fric > 0:
                    sectors.append({'start': sec.get('start', 0),
                                    'end': sec.get('end', 0),
                                    'friction': float(fric)})
            self.get_logger().info(f"[GGTuner] Friction sectors from yaml: {len(sectors)} sectors")
            return sectors
        except Exception as e:
            self.get_logger().warning(f"[GGTuner] friction_scaling.yaml parse error: {e}")
            return []


    def _replace_dir(self, dst, src):
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    ## IY : publish snapshot-name selectors to rosparam (consumed by velopt/FBGA)
    def _publish_sector_ggv_map(self, slot_overrides):
        for i in range(5):
            name = (slot_overrides.get(i, '') or '').strip()
            rospy.set_param(f'/gg_tuner/sector_ggv_map/sector{i}', name)
        set_slots = {i: slot_overrides[i] for i in range(5)
                     if (slot_overrides.get(i, '') or '').strip()}
        if set_slots:
            self.get_logger().info(f"[GGTuner] sector_ggv_map published: {set_slots}")
        else:
            self.get_logger().info("[GGTuner] sector_ggv_map cleared (all latest)")
    ## IY : end

    ## IY : raceline — passes safety_distance from rqt
    def _run_raceline(self, vehicle_name, map_name, safety_distance=0.20):
        if map_name not in self.available_maps:
            self.get_logger().error(f"[GGTuner] map '{map_name}' not found. "
                         f"Available: {self.available_maps}")
            self.status_pub.publish(f"FAILED_RACELINE: invalid map")
            return False
        self.status_pub.publish(f"RACELINE_STARTED: {vehicle_name}")
        self.get_logger().info(f"[GGTuner] [raceline] map={map_name}, vehicle={vehicle_name}, "
                      f"safety_distance={safety_distance}")
        cmd = [
            'roslaunch', 'stack_master', '3d_global_line.launch',
            f'map:={map_name}',
            f'vehicle:={vehicle_name}',
            f'safety_distance:={safety_distance}',
            'start_from:=5',
        ]
        ok = self._run_and_stream(cmd, tag='raceline', timeout=900)
        if ok:
            self.status_pub.publish(f"RACELINE_DONE: {vehicle_name}")
        else:
            self.status_pub.publish(f"FAILED_RACELINE: {vehicle_name}")
        return ok
    ## IY : end

    def _run_fbga_planner(self, vehicle_name, enable_mu=True, force_restart=False):
        self.get_logger().info(
            f"[GGTuner] [fbga] update: {vehicle_name}, enable_mu={enable_mu}, "
            f"force_restart={force_restart}")
        self.status_pub.publish(f"FBGA_STARTED: {vehicle_name}")

        gg_bin = os.path.join(
            self.data_path, 'gg_diagrams', vehicle_name,
            'velocity_frame', 'gg.bin')
        params_yml = os.path.join(
            self.data_path, 'vehicle_params',
            'params_' + vehicle_name + '.yml')
        if not os.path.exists(params_yml):
            self.get_logger().error(f"[GGTuner] params yml missing: {params_yml}")
            self.status_pub.publish(f"FAILED_FBGA: {vehicle_name}")
            return False

        rospy.set_param('/fbga_planner/gg_bin', gg_bin)
        rospy.set_param('/fbga_planner/params_yml', params_yml)
        rospy.set_param('/fbga_planner/enable_mu', bool(enable_mu))

        if not force_restart:
            try:
                rospy.wait_for_service('/fbga/reload', timeout=0.5)
                reload_srv = rospy.ServiceProxy('/fbga/reload', Trigger)
                resp = reload_srv()
                if resp.success:
                    self.get_logger().info(
                        f"[GGTuner] [fbga] hot-reloaded: {resp.message}")
                    self.status_pub.publish(f"FBGA_DONE: {vehicle_name}")
                    return True
                else:
                    self.get_logger().warning(
                        f"[GGTuner] [fbga] reload returned failure: "
                        f"{resp.message} → cold start")
            except (rospy.ROSException, rospy.ServiceException) as e:
                self.get_logger().info(
                    f"[GGTuner] [fbga] reload unavailable ({e}) → cold start")

        try:
            subprocess.run(['rosnode', 'kill', '/fbga_planner'],
                           capture_output=True, timeout=5, check=False)
            time.sleep(0.3)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if self.fbga_proc is not None and self.fbga_proc.poll() is None:
            try:
                self.fbga_proc.terminate()
                self.fbga_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.fbga_proc.kill()
        self.fbga_proc = None

        cmd = [
            'rosrun', 'stack_master', 'fbga_velocity_planner.py',
            '__name:=fbga_planner',
            f'_gg_bin:={gg_bin}',
            f'_params_yml:={params_yml}',
            f'_enable_mu:={str(enable_mu).lower()}',
        ]
        self.get_logger().info(f"[GGTuner] [fbga] cold start: {' '.join(cmd)}")
        try:
            self.fbga_proc = subprocess.Popen(cmd)
            return True
        except OSError as e:
            self.get_logger().error(f"[GGTuner] Failed to start FBGA: {e}")
            self.status_pub.publish(f"FAILED_FBGA: {vehicle_name}")
            return False

    def _kill_fbga_planner(self):
        self.get_logger().info("[GGTuner] [fbga] kill requested (checkbox off)")
        try:
            subprocess.run(['rosnode', 'kill', '/fbga_planner'],
                           capture_output=True, timeout=5, check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        if self.fbga_proc is not None and self.fbga_proc.poll() is None:
            try:
                self.fbga_proc.terminate()
                self.fbga_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.fbga_proc.kill()
        self.fbga_proc = None
        self.get_logger().info("[GGTuner] [fbga] killed")
    ## IY : end

    ## IY : publish GGV diamond results as JSON for rqt_gg_viewer
    def _publish_gg_results(self, vehicle_name, tuning=None, slope_results=None):
        vf_dir = os.path.join(self.data_path, 'gg_diagrams', vehicle_name,
                              'velocity_frame')
        if not os.path.isdir(vf_dir):
            self.get_logger().warning(f"[GGTuner] publish_gg_results: no velocity_frame dir: {vf_dir}")
            return
        try:
            v_list = np.load(os.path.join(vf_dir, 'v_list.npy')).tolist()
            g_list = np.load(os.path.join(vf_dir, 'g_list.npy')).tolist()
            ax_max = np.load(os.path.join(vf_dir, 'ax_max.npy')).tolist()
            ax_min = np.load(os.path.join(vf_dir, 'ax_min.npy')).tolist()
            ay_max = np.load(os.path.join(vf_dir, 'ay_max.npy')).tolist()
            gg_exp = np.load(os.path.join(vf_dir, 'gg_exponent.npy')).tolist()
        except (FileNotFoundError, OSError) as e:
            self.get_logger().warning(f"[GGTuner] publish_gg_results: npy load failed: {e}")
            return
        # auto-load slope results from 3D diamond arrays if available
        ## IY : honor enable_slope meta — stale *_3d.npy may exist when ggv was
        ##      regenerated with enable_slope=False (gen_diamond reuses slope_data/)
        enable_slope_meta = True  # default: trust file presence (legacy)
        es_path = os.path.join(vf_dir, 'enable_slope.npy')
        if os.path.isfile(es_path):
            try:
                enable_slope_meta = bool(np.load(es_path))
            except (FileNotFoundError, OSError, ValueError):
                pass
        ## IY : end
        if slope_results is None and enable_slope_meta:
            slope_list_path = os.path.join(vf_dir, 'slope_list_deg.npy')
            ax_max_3d_path = os.path.join(vf_dir, 'ax_max_3d.npy')
            if os.path.exists(slope_list_path) and os.path.exists(ax_max_3d_path):
                try:
                    slope_results = {
                        'slope_list_deg': np.load(slope_list_path).tolist(),
                        'ax_max': np.load(ax_max_3d_path).tolist(),
                        'ax_min': np.load(os.path.join(vf_dir, 'ax_min_3d.npy')).tolist(),
                        'ay_max': np.load(os.path.join(vf_dir, 'ay_max_3d.npy')).tolist(),
                        ## IY : add gg_exponent_3d for viewer slope-aware diamond
                        'gg_exponent': np.load(os.path.join(vf_dir, 'gg_exponent_3d.npy')).tolist(),
                    }
                    self.get_logger().info(f"[GGTuner] 3D slope diamond loaded for viewer")
                except (FileNotFoundError, OSError):
                    pass
        elif not enable_slope_meta:
            self.get_logger().info(f"[GGTuner] enable_slope=False → skip 3D slope payload "
                          f"(stale *_3d.npy ignored)")
        payload = {
            'vehicle_name': vehicle_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'v_list': v_list,
            'g_list': g_list,
            'diamond': {
                'ax_max': ax_max,
                'ax_min': ax_min,
                'ay_max': ay_max,
                'gg_exponent': gg_exp,
            },
            'tuning_params': tuning or {},
            'slope_results': slope_results,
        }
        self.results_pub.publish(json.dumps(payload))
        self.get_logger().info(f"[GGTuner] Published /gg_results "
                      f"(V={len(v_list)}, g={len(g_list)})")
    ## IY : end

    def _run_full_pipeline(self, tuning, run_opts):
        with self.pipeline_lock:
            try:
                self.get_logger().info(f"[GGTuner] ===== Pipeline start =====")
                self.get_logger().info(f"[GGTuner] tuning: {tuning}")
                self.get_logger().info(f"[GGTuner] options: {run_opts}")
                self.status_pub.publish(f"STARTED: {self.base_vehicle}")

                vehicle_name = f"{self.base_vehicle}_latest"

                if not run_opts['run_ggv']:
                    ## IY : raceline_ggv override — activate snapshot into _latest
                    if run_opts['raceline_ggv']:
                        if not self._activate_snapshot_as_latest(
                                run_opts['raceline_ggv']):
                            self.status_pub.publish(
                                f"FAILED: snapshot_missing:{run_opts['raceline_ggv']}")
                            return
                    ## IY : end
                    latest_gg = os.path.join(
                        self.data_path, 'gg_diagrams', vehicle_name)
                    if not os.path.exists(latest_gg):
                        self.get_logger().error(
                            f"[GGTuner] latest gg_diagrams missing: {latest_gg} "
                            f"(run_ggv=False)")
                        self.status_pub.publish("FAILED: no latest")
                        return
                    self.get_logger().info(
                        f"[GGTuner] Stage 2 SKIP (run_ggv=False), reusing latest")
                    self.status_pub.publish(f"GGV_SKIP: {vehicle_name}")
                    ## IY : publish existing results for rqt_gg_viewer
                    self._publish_gg_results(vehicle_name, tuning)
                    ## IY : end
                else:
                    self.get_logger().info(
                        f"[GGTuner] Stage 2: compute fresh into latest "
                        f"(friction={tuning.get('friction', '?')})")
                    merged = self._merge_all_params(tuning)
                    self._write_latest_params_yml(merged)
                    ok = self._run_fast_ggv(
                        vehicle_name,
                        full_resolution=run_opts['full_resolution'],
                        enable_slope=run_opts.get('enable_slope', False),
                        slope_max_deg=run_opts.get('slope_max_deg', 15.0),
                        slope_N=run_opts.get('slope_N', 5),
                        slope_ax_scale=run_opts.get('slope_ax_scale', 1.0),
                        slope_normal_scale=run_opts.get('slope_normal_scale', 1.0))
                    if not ok:
                        self.status_pub.publish(f"FAILED_GGV: {vehicle_name}")
                        return
                    if not self._copy_to_gg_diagrams(vehicle_name):
                        self.status_pub.publish(f"FAILED_GGV: {vehicle_name}")
                        return
                    self._save_meta(vehicle_name, tuning)
                    ## IY : save ggv meta for downstream auto-pairing
                    self._save_ggv_meta(
                        vehicle_name,
                        slope_ax_scale=run_opts.get('slope_ax_scale', 1.0),
                        enable_slope=run_opts.get('enable_slope', False))
                    ## IY : end
                    self.status_pub.publish(f"GGV_DONE: {vehicle_name}")
                    ## IY : publish fresh results for rqt_gg_viewer
                    self._publish_gg_results(vehicle_name, tuning)
                    ## IY : end
                ## IY : end
                    
                # Publish sector snapshot-name selectors for velopt/FBGA
                slot_overrides = {
                    i: run_opts[f'ggv_sector{i}'] for i in range(5)
                }
                self._publish_sector_ggv_map(slot_overrides)

                # ---- Stage 3: raceline (optional) ----
                if run_opts['regen_raceline']:
                    ## IY : overlay rqt RACELINE_KEYS onto latest yml (idempotent)
                    self._update_raceline_keys_in_yml(vehicle_name, tuning)
                    ## IY : end
                    ok = self._run_raceline(
                        vehicle_name, run_opts['map'],
                        safety_distance=run_opts['safety_distance'])
                    if not ok:
                        self.get_logger().error(f"[GGTuner] raceline failed")
                        return
                else:
                    self.get_logger().info(f"[GGTuner] raceline regen SKIP")

                if run_opts['run_fbga']:
                    force_restart = bool(run_opts['regen_raceline'])
                    ok = self._run_fbga_planner(
                        vehicle_name,
                        enable_mu=run_opts['enable_mu'],
                        force_restart=force_restart)
                    if not ok:
                        return
                    self.status_pub.publish(f"DONE_ALL: {vehicle_name}")
                else:
                    self._kill_fbga_planner()
                    self.get_logger().info(f"[GGTuner] fbga killed (run_fbga=False)")
                    self.status_pub.publish(f"DONE_FBGA_OFF: {vehicle_name}")
                ## IY : end

                self.get_logger().info(f"[GGTuner] ===== Pipeline done: {vehicle_name} =====")

            except Exception as e:
                self.get_logger().error(f"[GGTuner] Pipeline exception: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                self.status_pub.publish(f"EXCEPTION: {str(e)[:100]}")
    ## IY : end
                
    def reconfigure_cb(self, config, level):

        if getattr(config, 'save_now', False):
            if self.pipeline_thread is not None and self.pipeline_thread.is_alive():
                self.get_logger().warning(
                    "[GGTuner] SAVE ignored: pipeline running "
                    " ")
            else:
                try:
                    self._snapshot_latest_to_version()
                except Exception as e:
                    self.get_logger().error(f"[GGTuner] SAVE exception: {e}")
                    self.status_pub.publish(f"SAVE_EXCEPTION: {str(e)[:80]}")
            config.save_now = False
            if not config.apply:
                return config
        ## IY : end

        if not config.apply:
            return config

        ## IY : refuse concurrent runs
        if self.pipeline_thread is not None and self.pipeline_thread.is_alive():
            self.get_logger().warning("[GGTuner] Pipeline already running — ignoring apply")
            config.apply = False
            return config
        ## IY : end

        # collect ALL tuning parameters
        tuning = {k: config[k] for k in self.ALL_TUNING_KEYS}

        run_opts = {
            'run_ggv':          bool(config.run_ggv),
            'full_resolution':  bool(config.full_resolution),
            'regen_raceline':   bool(config.regen_raceline),
            'map':              str(config.map),
            'run_fbga':         bool(config.run_fbga),
            'enable_mu':        bool(config.enable_mu),
            'safety_distance':  float(config.safety_distance),
            ## IY : per-sector snapshot selectors (empty=latest) for velopt/FBGA
            'ggv_sector0':      str(getattr(config, 'ggv_sector0', '')).strip(),
            'ggv_sector1':      str(getattr(config, 'ggv_sector1', '')).strip(),
            'ggv_sector2':      str(getattr(config, 'ggv_sector2', '')).strip(),
            'ggv_sector3':      str(getattr(config, 'ggv_sector3', '')).strip(),
            'ggv_sector4':      str(getattr(config, 'ggv_sector4', '')).strip(),
            ## IY : raceline-dedicated snapshot (empty=latest)
            'raceline_ggv':     str(getattr(config, 'raceline_ggv', '')).strip(),
            ## IY : slope sweep for rqt_gg_viewer
            'enable_slope':     bool(getattr(config, 'enable_slope', False)),
            'slope_max_deg':    float(getattr(config, 'slope_max_deg', 15.0)),
            'slope_N':          int(getattr(config, 'slope_N', 5)),
            'slope_ax_scale':   float(getattr(config, 'slope_ax_scale', 1.0)),
            'slope_normal_scale': float(getattr(config, 'slope_normal_scale', 1.0)),
            ## IY : end
        }
        ## IY : end

        ## IY : validate map
        if run_opts['regen_raceline'] and run_opts['map'] not in self.available_maps:
            self.get_logger().error(f"[GGTuner] Invalid map '{run_opts['map']}'. "
                         f"Available: {self.available_maps}")
            self.status_pub.publish(f"FAILED_RACELINE: invalid map")
            config.apply = False
            return config
        ## IY : end

        ## IY : background thread
        self.pipeline_thread = threading.Thread(
            target=self._run_full_pipeline,
            args=(tuning, run_opts),
            daemon=True)
        self.pipeline_thread.start()
        self.get_logger().info("[GGTuner] Pipeline spawned in background thread")
        ## IY : end

        config.apply = False
        return config

    def _shutdown_cleanup(self):
        if self.fbga_proc is not None and self.fbga_proc.poll() is None:
            self.get_logger().info("[GGTuner] Terminating FBGA subprocess...")
            try:
                self.fbga_proc.terminate()
                self.fbga_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.fbga_proc.kill()


if __name__ == '__main__':
    rospy.init_node("gg_tuner")
    node = GGTunerNode()
    rospy.spin()
