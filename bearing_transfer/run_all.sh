#!/bin/bash
set -euo pipefail
DATA_ROOT=${1:-./data}
PYTHONPATH=$(pwd) python scripts/prepare_features.py --data_root "$DATA_ROOT"
PYTHONPATH=$(pwd) python scripts/train_source.py
PYTHONPATH=$(pwd) python scripts/transfer_and_predict.py
PYTHONPATH=$(pwd) python scripts/explainability.py
PYTHONPATH=$(pwd) python scripts/make_report.py
