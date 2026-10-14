#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv .venv-cloud
source .venv-cloud/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cloud.txt
python -m pip install -e . --no-deps
python -m pip check
python -m pip freeze > requirements-cloud.resolved.txt
python -m robot_data_lab doctor
