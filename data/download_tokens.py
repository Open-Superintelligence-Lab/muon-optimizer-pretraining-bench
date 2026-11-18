"""Fetch the GPT-2-token FineWeb slices of the NANOGPT-OPT benchmark and check their sha256.

Source: Hugging Face dataset kjj0/fineweb10B-gpt2 at commit 889765ea1f903759787add96995d81171b632d0c,
the exact data of the modded-nanogpt speedrun and its optimization track (KellerJordan/modded-nanogpt,
data/cached_fineweb10B.py). FineWeb is released under ODC-By 1.0.

Laptop subset (default):
  tok_train.npy = first 20,000,000 tokens of fineweb_train_000001.bin   (40 MB, uint16)
  tok_val.npy   = first  1,048,576 tokens of fineweb_val_000000.bin     ( 2 MB, uint16)
Each .bin has a 1,024-byte header (256 int32, magic 20240520, version 1, token count); only the
needed byte range is fetched with HTTP Range.

Full set (for a GPU): `python3 -I data/download_tokens.py --full N` fetches the whole val
shard and train shards 1..N (100M tokens each, 200 MB) as .bin files into data/fineweb10B/, the
layout train_gpt_simple.py expects. Full shards are checked only by header and size.

Downloads are untrusted data: written only into this folder (or OPT_PRETRAIN_DATA); run with python3 -I.
"""
import hashlib
import os
import sys
import urllib.request

import numpy as np

HERE = os.environ.get('OPT_PRETRAIN_DATA') or os.path.dirname(os.path.abspath(__file__))
REV = '889765ea1f903759787add96995d81171b632d0c'
URL = 'https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/' + REV + '/{}'
HEADER = 1024
FILES = {  # out name: (source file, tokens, sha256 of the saved .npy)
    'tok_train.npy': ('fineweb_train_000001.bin', 20_000_000,
                      '5b638261d3ccf12add65071f400169d585ec2eb57f71a74e8ee9d521f859927c'),
    'tok_val.npy': ('fineweb_val_000000.bin', 1_048_576,
                    'b1880fe863379575b2cf7c2a607ecd64b109f104cdc7e74eb47419bbb975cf93'),
}


def fetch(name, n_tok):
    req = urllib.request.Request(URL.format(name), headers={'Range': f'bytes=0-{HEADER + 2 * n_tok - 1}'})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    head = np.frombuffer(raw[:HEADER], dtype=np.int32)
    if head[0] != 20240520 or head[1] != 1:
        raise SystemExit(f'{name}: bad header {head[:3]}')
    tok = np.frombuffer(raw[HEADER:HEADER + 2 * n_tok], dtype=np.uint16).copy()
    if len(tok) != n_tok:
        raise SystemExit(f'{name}: got {len(tok)} tokens, expected {n_tok}')
    return tok


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def full(n):
    out = os.path.join(HERE, 'fineweb10B')
    os.makedirs(out, exist_ok=True)
    for name in ['fineweb_val_000000.bin'] + ['fineweb_train_%06d.bin' % i for i in range(1, n + 1)]:
        p = os.path.join(out, name)
        if not os.path.exists(p):
            urllib.request.urlretrieve(URL.format(name), p + '.part')
            os.replace(p + '.part', p)
        head = np.fromfile(p, dtype=np.int32, count=3)
        ok = head[0] == 20240520 and os.path.getsize(p) == HEADER + 2 * int(head[2])
        print(name, 'OK' if ok else 'BAD', int(head[2]), 'tokens')


def main(argv):
    if '--full' in argv:
        return full(int(argv[argv.index('--full') + 1]))
    bad = 0
    for out, (name, n_tok, want) in FILES.items():
        path = os.path.join(HERE, out)
        if not os.path.exists(path) or (want and sha(path) != want):
            np.save(path + '.new.npy', fetch(name, n_tok))
            got = sha(path + '.new.npy')
            if want and got != want:
                print(out, 'CHECKSUM MISMATCH', got, '(left as', out + '.new.npy)'); bad += 1; continue
            os.replace(path + '.new.npy', path)
        print(out, 'OK', sha(path))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main(sys.argv[1:])
