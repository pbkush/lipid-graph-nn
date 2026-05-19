#!/bin/bash
# rsync a preprocessed-graphs zip (built by scripts/training/preprocess_graphs.py)
# to the Goethe HPC cluster. Pass the run name (= property-set folder name) as $1
# or via $RUN_NAME.

set -euo pipefail

cd "$(dirname "$0")/../.."

USER=pberger
GROUP="$(python scripts/python/print_config_var.py hpc.group)"
WORK_SUBPATH="$(python scripts/python/print_config_var.py hpc.work_subpath)"
PARENT_DIR="$(python scripts/python/print_config_var.py paths.preprocessed_graphs_dir)"

RUN_NAME="${1:-${RUN_NAME:?pass the run name as \$1 or set RUN_NAME}}"
ZIP_PATH="$PARENT_DIR/archives/${RUN_NAME}.zip"

rsync -avh --partial --progress \
  "$ZIP_PATH" \
  "$USER@goethe.hhlr-gu.de:/work/$GROUP/$USER/$WORK_SUBPATH/preprocessed_graphs/archives/"

#rsync -avh --partial --progress \
#  results/properties/ \
#  "$USER@goethe.hhlr-gu.de:/work/$GROUP/$USER/$WORK_SUBPATH/results/properties/"
