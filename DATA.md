# Data

Nothing in `data/` is committed except the download scripts. One command fetches everything the laptop presets need, pinned and checksum-checked:

```bash
pip install numpy tiktoken
sh data/download.sh            # NANOGPT-OPT laptop tokens (42 MB)
sh data/download.sh full 20    # also the official NANOGPT-OPT set: val shard + 20 train shards (about 4 GB)
```

## NANOGPT-OPT (main benchmark): `data/download_tokens.py`

| File | Content | Size | sha256 |
|---|---|---|---|
| `data/tok_train.npy` | first 20,000,000 GPT-2 tokens of `fineweb_train_000001.bin` (uint16) | 40 MB | `5b638261d3ccf12add65071f400169d585ec2eb57f71a74e8ee9d521f859927c` |
| `data/tok_val.npy` | first 1,048,576 GPT-2 tokens of `fineweb_val_000000.bin` (uint16) | 2 MB | `b1880fe863379575b2cf7c2a607ecd64b109f104cdc7e74eb47419bbb975cf93` |

- Source: Hugging Face dataset `kjj0/fineweb10B-gpt2`, pinned to commit `889765ea1f903759787add96995d81171b632d0c`. These are the files the modded-nanogpt speedrun and its optimization track train on (`data/cached_fineweb10B.py` in KellerJordan/modded-nanogpt).
- Only the needed byte range of each `.bin` is fetched with an HTTP Range request; the 1,024-byte header (magic 20240520, version 1) is checked before the tokens are kept.
- The laptop preset trains on the first 204,800 tokens of `tok_train.npy`, in order, and validates on the first 32,768 tokens of `tok_val.npy`; the rest of the slice is there for longer laptop runs.
- `--full N` writes the whole val shard and train shards 1..N as `.bin` files into `data/fineweb10B/`, the layout `env/nanogpt_train.py preset=full` reads (shards 1-20, val = first 10,485,760 tokens, as in track 3). Full shards are checked by header and size only.
- Checksums verified on 2026-11-18 by a fresh download.

## Notes

- License: FineWeb is released under ODC-By 1.0 and is subject to Common Crawl's terms of use.
- Downloaded data is untrusted: it lives only in `data/`, and every script that reads it is meant to be run with `python3 -I`.
- To keep data elsewhere, set `OPT_PRETRAIN_DATA=/path/to/dir`.
