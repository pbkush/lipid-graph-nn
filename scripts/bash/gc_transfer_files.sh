#!/bin/bash
# rsync the preprocessed colab bundle zip to the Goethe HPC cluster.

set -euo pipefail

cd "$(dirname "$0")/../.."

USER="${USER:-pberger}"
GROUP="$(python scripts/python/print_config_var.py hpc.group)"
WORK_SUBPATH="$(python scripts/python/print_config_var.py hpc.work_subpath)"
BUNDLE_DIR="$(python scripts/python/print_config_var.py paths.subset_bundle_dir)"
BUNDLE_NAME="$(basename "$BUNDLE_DIR")"

rsync -avh --partial --progress \
  "${BUNDLE_NAME}.zip" \
  "$USER@goethe.hhlr-gu.de:/work/$GROUP/$USER/$WORK_SUBPATH/"

#rsync -avh --partial --progress \
#  results/properties/ \
#  "$USER@goethe.hhlr-gu.de:/work/$GROUP/$USER/$WORK_SUBPATH/results/properties/"
