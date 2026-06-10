#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# port_to_realcar.sh — sync THIS repo's nonlinear_mpc_acados package onto the
# real-car (Mac/NUC) workspace copy, WITHOUT touching any sibling package.
#
# Target (T7 ws): .../src/IFAC2026_SH/src/control/nonlinear_mpcc/nonlinear_mpc_acados
# The target may be on T7 or wherever T7 is remounted — pass its path as $1.
#
# SAFETY:
#   * Dry-run by DEFAULT. Nothing is written unless you pass --apply.
#   * rsync source = our package, dest = the target package dir → changes are
#     inherently scoped to that one directory. Sibling packages
#     (calibration/sensor/slam/system/…) are never visited.
#   * Refuses to run unless the target looks like the nonlinear_mpc_acados
#     package (basename + package.xml/setup.py sanity check).
#   * Build/cache artifacts are excluded so we never ship host binaries.
#
# Usage:
#   scripts/port_to_realcar.sh <TARGET_PKG_DIR>            # dry-run (preview)
#   scripts/port_to_realcar.sh <TARGET_PKG_DIR> --apply    # actually sync
#
# Example (when T7 is mounted here):
#   scripts/port_to_realcar.sh \
#     /media/hmcl/T7/ros2_ws/ros2_ws/src/IFAC2026_SH/src/control/nonlinear_mpcc/nonlinear_mpc_acados
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SRC_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_PKG="$SRC_REPO/src/nonlinear_mpc_acados"
ALGO_DOC="$SRC_REPO/docs/MPCC_ALGORITHM_REFERENCE.md"
HANDOFF_DOC="$SRC_REPO/docs/SESSION_HANDOFF_trackB_2026-06-09.md"

# Parse args in a loop so flags work in ANY position (not positional-only).
TARGET=""
APPLY=""
WITH_CONFIG=""        # config/ is car-specific tuning → excluded unless opted in
for a in "$@"; do
  case "$a" in
    --apply)       APPLY="--apply" ;;
    --with-config) WITH_CONFIG=1 ;;
    -h|--help)     TARGET="" ;;     # fall through to usage
    --*)           echo "ERROR: unknown flag: $a" >&2; exit 2 ;;
    *)             TARGET="$a" ;;
  esac
done

if [[ -z "$TARGET" ]]; then
  echo "usage: $0 <TARGET_nonlinear_mpc_acados_dir> [--apply] [--with-config]" >&2
  echo "  default          = dry-run preview (nothing written)" >&2
  echo "  --apply          = actually sync" >&2
  echo "  --with-config    = ALSO overwrite config/ (car-specific tuning!); off by default" >&2
  exit 2
fi

# ── sanity: source ──────────────────────────────────────────────────────────
[[ -d "$SRC_PKG" ]] || { echo "ERROR: source package not found: $SRC_PKG" >&2; exit 1; }

# ── sanity: target must be THE package (avoid clobbering the wrong dir) ──────
if [[ "$(basename "$TARGET")" != "nonlinear_mpc_acados" ]]; then
  echo "ERROR: target basename is not 'nonlinear_mpc_acados': $TARGET" >&2
  echo "       refusing — point me at the package dir, not its parent." >&2
  exit 1
fi
if [[ ! -d "$TARGET" ]]; then
  echo "ERROR: target dir does not exist (is T7 mounted?): $TARGET" >&2
  exit 1
fi
if [[ ! -f "$TARGET/package.xml" && ! -f "$TARGET/setup.py" ]]; then
  echo "ERROR: target has no package.xml/setup.py — does not look like a ROS pkg: $TARGET" >&2
  echo "       refusing to write." >&2
  exit 1
fi

# ── what NOT to ship (host build artifacts, caches) ─────────────────────────
EXCLUDES=(
  --exclude '__pycache__/'   --exclude '*.pyc'
  --exclude '.pytest_cache/' --exclude '*.egg-info/'
  --exclude 'build/'         --exclude 'install/'   --exclude 'log/'
  --exclude '.git/'
)
# config/ holds CAR-SPECIFIC tuning (use_dynamic/use_lmpc/max_speed/a_lat/mu).
# Excluded by default so we never silently clobber the car's calibration; pass
# --with-config to overwrite it deliberately.
if [[ -z "$WITH_CONFIG" ]]; then
  EXCLUDES+=(--exclude 'config/')
fi
echo "════════════════════════════════════════════════════════════════════"
echo " PORT  nonlinear_mpc_acados"
echo "   from : $SRC_PKG"
echo "   to   : $TARGET"
echo "   git  : $(git -C "$SRC_REPO" rev-parse --short HEAD 2>/dev/null || echo '?') on $(git -C "$SRC_REPO" branch --show-current 2>/dev/null || echo '?')"
echo "   mode : $([[ "$APPLY" == "--apply" ]] && echo 'APPLY (writing)' || echo 'DRY-RUN (preview only)')"
echo "════════════════════════════════════════════════════════════════════"

RSYNC_FLAGS=(-a --itemize-changes "${EXCLUDES[@]}")
if [[ "$APPLY" != "--apply" ]]; then
  RSYNC_FLAGS+=(--dry-run)
fi

# trailing slash on source = copy CONTENTS into target dir
rsync "${RSYNC_FLAGS[@]}" "$SRC_PKG/" "$TARGET/"

# ── drop the docs alongside the package ─────────────────────────────────────
if [[ "$APPLY" == "--apply" ]]; then
  mkdir -p "$TARGET/docs"
  [[ -f "$ALGO_DOC" ]]    && cp "$ALGO_DOC"    "$TARGET/docs/"
  [[ -f "$HANDOFF_DOC" ]] && cp "$HANDOFF_DOC" "$TARGET/docs/"
fi

echo
echo "──────────────────────────────────────────────────────────────────────"
if [[ "$APPLY" != "--apply" ]]; then
  echo "DRY-RUN complete. Re-run with --apply to write the above changes."
else
  echo "APPLY complete. NEXT on the real car:"
  echo "  1. acados REBUILD on that platform (Mac=ARM rebuild, CUDA forbidden; NUC=x86)."
  echo "     Clear stale codegen first:  rm -rf /tmp/acados_codegen_evompcc*"
  echo "  2. colcon build --packages-select nonlinear_mpc_acados ; source install/setup.bash"
  if [[ -n "$WITH_CONFIG" ]]; then
    echo "  3. ⚠ config/ WAS OVERWRITTEN (--with-config). Re-check car-specific tuning:"
    echo "       config/ddrx_unified_params.yaml  (use_dynamic / use_lmpc / max_speed / a_lat / mu-match)"
  else
    echo "  3. config/ was NOT synced (car-specific tuning preserved). If you need new"
    echo "       params, merge them by hand or re-run with --with-config."
  fi
  echo "  4. Per car: low-grip→dynamic(+mu match), high-grip→kinematic(LMPC auto-off)."
fi
echo "  ⚠ sibling packages under .../control/nonlinear_mpcc/ and elsewhere were NOT touched."
echo "──────────────────────────────────────────────────────────────────────"
