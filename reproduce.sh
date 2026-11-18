#!/usr/bin/env bash
# Rebuild the NANOGPT-OPT baseline table: data, official code at pinned commits, match checks,
# 5 optimizers x 2-3 learning rates, manifest, table. About 1-1.5 hours with 2 jobs on a MacBook CPU.
set -euo pipefail
cd "$(dirname "$0")"
sh data/download.sh                                   # tok_train/tok_val.npy, sha256-checked
sh third_party/fetch.sh                               # official repos at pinned commits (gitignored)
python3 -I tests/check_official.py                    # vendored code == upstream; same loss and updates
python3 -I experiments/nanogpt_baselines.py --jobs 2  # -> experiments/runs/NANOGPT-OPT.jsonl
python3 -I experiments/make_manifest.py               # baselines/manifest.json
python3 benchmark/table.py
