#!/usr/bin/env bash
# Build acados + its Python interface for the nonlinear_mpc_acados node.
#
# Defaults to ~/acados as the install location. Override via:
#   ACADOS_DIR=/opt/acados ./install_acados.sh
#
# What it does:
#   1. clone (or update) the acados source
#   2. cmake + make install (BLASFEO + HPIPM + libacados.so)
#   3. pip install the acados_template Python interface
#   4. print the env-var lines you should add to ~/.bashrc
#
# After this script, `mpc_backend:=acados` works.
set -euo pipefail

ACADOS_DIR="${ACADOS_DIR:-$HOME/acados}"
JOBS="${JOBS:-$(nproc)}"

echo "[acados] target dir: $ACADOS_DIR"

if [ -d "$ACADOS_DIR/.git" ]; then
    echo "[acados] existing checkout — pulling"
    git -C "$ACADOS_DIR" pull --ff-only
else
    echo "[acados] cloning"
    git clone https://github.com/acados/acados.git "$ACADOS_DIR"
fi

cd "$ACADOS_DIR"
git submodule update --recursive --init

mkdir -p build && cd build
cmake -DACADOS_INSTALL_DIR="$ACADOS_DIR" -DACADOS_PYTHON=ON ..
make install -j"$JOBS"

cd "$ACADOS_DIR"
echo "[acados] installing Python interface"
# --break-system-packages: Ubuntu 24.04 PEP 668. Use a venv if you prefer.
pip install -e "interfaces/acados_template/" --break-system-packages \
    || pip install -e "interfaces/acados_template/"

cat <<EOF

------------------------------------------------------------
[acados] build complete.

Add the following to your ~/.bashrc (then 'source ~/.bashrc'):

    export ACADOS_SOURCE_DIR=$ACADOS_DIR
    export LD_LIBRARY_PATH=\$ACADOS_SOURCE_DIR/lib:\$LD_LIBRARY_PATH

The launch file (mpc.launch.py) reads ACADOS_SOURCE_DIR and prepends
\$ACADOS_SOURCE_DIR/lib to LD_LIBRARY_PATH automatically when running
through ros2 launch — but you still need ACADOS_SOURCE_DIR exported
for any direct 'python3 -c "from acados_template import ..."' use.
------------------------------------------------------------
EOF
