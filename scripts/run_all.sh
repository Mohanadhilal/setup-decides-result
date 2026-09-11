#!/usr/bin/env bash
# Reproduces every number in the paper, in dependency order.
# Usage:  bash scripts/run_all.sh [workers]        (default 5)
set -euo pipefail
W="${1:-5}"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"

echo "== 0. sanity check (no training) =="
python scripts/train_multiseed.py --mock

echo "== 1. algorithm comparison, balanced preference  -> Table 7 =="
python scripts/train_multiseed.py --algos SAC TD3 PPO --prefs 2 --seeds 0 1 2 3 4 --workers "$W"

echo "== 2. remaining preferences                      -> Table 3 =="
python scripts/train_multiseed.py --algos SAC --prefs 0 1 3 4 5 --seeds 0 1 2 3 4 --workers "$W"

echo "== 3. protocol re-selection, classical grid, matched comparison -> Tables 3,4,5,7 =="
python scripts/reeval_protocol.py --stage all --workers "$W"

echo "== 4. safety-gate strictness sweep               -> Table 9 =="
python scripts/reselect_gate_sweep.py --algos SAC --workers "$W"

echo "== 5. step disturbances on the test seeds        -> Table 8 =="
python scripts/robustness_protocol.py --pref 2 --workers "$W"

echo "== 6. seed-set offset control                    -> Section 4.4 =="
python scripts/eval_classical_seedsets.py

echo "== 7. figures                                    -> Figures 2, 3 =="
python scripts/make_figures.py

echo "done. Raw outputs are under results/."
