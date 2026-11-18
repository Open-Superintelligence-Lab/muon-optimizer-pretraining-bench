#!/bin/sh
# Clone the official repos at the pinned commits into third_party/ (gitignored, untrusted: read, never run).
# tests/check_official.py compares baselines/official/ and env/nanogpt_arch.py with these clones.
set -e
cd "$(dirname "$0")"
get() { [ -d "$2" ] || { git init -q "$2"; git -C "$2" fetch -q --depth 1 "https://github.com/$1.git" "$3"; git -C "$2" checkout -q FETCH_HEAD; }; echo "$2 $(git -C "$2" rev-parse HEAD)"; }
get KellerJordan/Muon Muon f98f1cacc0263b04290753e32be8d498c1efc806
get zichongli5/NorMuon NorMuon c6989a8354730695d9f5a9faa6c55eeb24865209
get nikhilvyas/SOAP SOAP a1e553530fde97d0e6b307d7c82ac6d38b072340
get MoonshotAI/Moonlight Moonlight c2ad5b20c605086526a179d36901bfc41b52b44b
get KellerJordan/modded-nanogpt modded-nanogpt 4ea6b937337a4889b8cfe3f38a93d120048d8f71
