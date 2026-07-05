#!/bin/zsh
set -euo pipefail
BUNDLE_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$BUNDLE_ROOT"
echo "==> Step 1/4: bootstrapping local Python environment"
./bootstrap_python_env.sh
echo ""
echo "==> Step 2/4: installing the bundle-local Jupyter kernel"
./install_repo_kernel.sh
echo ""
echo "==> Step 3/4: restoring R packages with renv"
R -q -e 'renv::restore(project = "semantic-change/workspace", lockfile = "semantic-change/workspace/renv.lock")'
echo ""
echo "==> Step 4/4: materializing repo prerequisites"
./.venv/bin/python semantic-change/workspace/code/scripts/setup_scibert_model.py
echo ""
echo "Repository configuration complete."
echo "Next: read the bundle README and the workspace README for the rebuild order."
