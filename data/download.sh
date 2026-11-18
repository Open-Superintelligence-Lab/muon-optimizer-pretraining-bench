#!/bin/sh
# Fetch all benchmark data into data/ (gitignored), pinned and checksum-checked. Untrusted: python3 -I.
#   sh data/download.sh            laptop subset: NANOGPT-OPT tokens (42 MB)
#   sh data/download.sh full 20    also the official NANOGPT-OPT set: val shard + 20 train shards (4 GB)
set -e
cd "$(dirname "$0")/.."
python3 -I data/download_tokens.py
if [ "$1" = full ]; then python3 -I data/download_tokens.py --full "${2:-20}"; fi
