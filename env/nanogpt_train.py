"""NANOGPT-OPT: the modded-nanogpt optimization track (records/track_3_optimization/train_gpt_simple.py
in KellerJordan/modded-nanogpt at 4ea6b937) with the matrix optimizer swapped between the official
Muon, NorMuon, Moonlight Muon and SOAP code in baselines/official/, plus an Adam reference.

What is kept from the official script: the GPT (env/nanogpt_arch.py, copied unchanged), the init,
the aux Adam groups for embedding / head / 1-D params (lr .7 / .004 / .015, betas (.8, .95),
eps 1e-10, decoupled weight decay .001), the stable-then-linear-decay schedule (cooldown .7), the
sequential data order from fineweb_train_000001.bin, summed-loss micro-batch accumulation, and
validation loss = mean cross-entropy on the first val_tokens tokens of fineweb_val_000000.bin.

Laptop subset (default, `preset=laptop`): 4 layers, width 128, context 256, 4 sequences
(1,024 tokens) per step, 200 steps (204,800 tokens), 32,768 validation tokens, fp32 on CPU, about
1.2-1.5 GB peak memory and about 7 minutes per run (two runs in parallel on a loaded MacBook). `preset=full` is the official size (12 layers, width 768, context
1,024, 512 sequences per step, 3,250 steps, 10,485,760 validation tokens) for runs on a GPU;
it needs `python3 -I data/download_tokens.py --full 20`.

    python3 -I env/nanogpt_train.py method=muon lr=0.025 wd=0.05
    python3 -I env/nanogpt_train.py method=soap lr=0.003 out=experiments/runs/NANOGPT.jsonl
"""
import json
import os
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'baselines', 'official'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env.nanogpt_arch import GPT  # noqa: E402

PRESETS = {
    'laptop': dict(layers=4, dim=128, seq=256, seqs=4, mbs=2, steps=200, val_tokens=32768, threads=2),
    'full': dict(layers=12, dim=768, seq=1024, seqs=512, mbs=64, steps=3250, val_tokens=20 * 524288, threads=0),
}
# official or paper defaults for the matrix optimizer of each method
METHODS = {
    'muon': dict(lr=0.025, wd=0.05),        # track-3 tuned baseline (#36)
    'normuon': dict(lr=0.035, wd=0.025),    # track-3 NorMuon (#10)
    'moonlight': dict(lr=0.004, wd=0.1),    # Moonlight toy_train.py default wd; lr is in Adam units
    'soap': dict(lr=0.003, wd=0.01),        # SOAP README
    'adam': dict(lr=0.0015, wd=0.0),        # track-3 Adam reference (#2)
}
VOCAB = 50304
DATA = os.environ.get('OPT_PRETRAIN_DATA') or os.path.join(ROOT, 'data')


def parse(argv):
    a = dict(method='muon', preset='laptop', cooldown=0.7, out='')
    for kv in argv:
        k, v = kv.split('=', 1)
        a[k] = v
    p = dict(PRESETS[a['preset']])
    p.update(METHODS[a['method']])
    for k, v in a.items():
        if k in ('method', 'preset', 'out'):
            p[k] = v
        else:
            p[k] = type(p.get(k, 0.0))(v) if k in p else float(v)
    return p


def load_tokens(c):
    if c['preset'] == 'laptop':
        tr, va = np.load(os.path.join(DATA, 'tok_train.npy')), np.load(os.path.join(DATA, 'tok_val.npy'))
    else:
        def shard(n):
            return np.fromfile(os.path.join(DATA, 'fineweb10B', n), dtype=np.uint16, offset=1024)
        va = shard('fineweb_val_000000.bin')
        tr = np.concatenate([shard('fineweb_train_%06d.bin' % i) for i in range(1, 21)])
    return torch.from_numpy(tr.astype(np.int64)), torch.from_numpy(va.astype(np.int64))


def init_(model):
    """Official init, train_gpt_simple.py lines 274-288."""
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith('weight'):
            if 'proj' in name:
                w.zero_()
            elif 'embed' in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
        elif name.endswith('bias'):
            w.zero_()
        elif name.endswith('gains'):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f'Uninitialized parameter: {name}')


def build_optimizers(model, c):
    aux = torch.optim.AdamW([dict(params=[model.embed.weight], lr=0.7),
                             dict(params=[model.proj.weight], lr=0.004),
                             dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)],
                            betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001)
    mats = [p for p in model.blocks.parameters() if p.ndim >= 2]
    m, lr, wd = c['method'], c['lr'], c['wd']
    if m == 'muon':
        from muon import SingleDeviceMuon
        opt = SingleDeviceMuon(mats, lr=lr, weight_decay=wd, momentum=0.95)
    elif m == 'normuon':
        from normuon import SingleDeviceNorMuon
        opt = SingleDeviceNorMuon(mats, lr=lr, weight_decay=wd, momentum=0.95, beta2=0.95)
    elif m == 'moonlight':
        from moonlight_muon import Muon as MoonMuon
        opt = MoonMuon(lr=lr, wd=wd, muon_params=mats, momentum=0.95, nesterov=True, ns_steps=5, adamw_params=[])
    elif m == 'soap':
        from soap import SOAP
        opt = SOAP(mats, lr=lr, betas=(0.95, 0.95), weight_decay=wd, precondition_frequency=10)
    elif m == 'adam':
        opt = torch.optim.Adam(mats, lr=lr, betas=(0.9, 0.95), eps=1e-10)
    else:
        raise SystemExit('unknown method ' + m)
    opts = [aux, opt]
    assert set(p for o in opts for g in o.param_groups for p in g['params']) == set(model.parameters())
    for o in opts:
        for g in o.param_groups:
            g['initial_lr'] = g['lr']
    return opts


@torch.no_grad()
def evaluate(model, va, c):
    n, T, mbs = c['val_tokens'], c['seq'], c['mbs']
    x, y = va[:n].view(-1, T), va[1:n + 1].view(-1, T)
    tot = 0.0
    for i in range(0, len(x), mbs):
        tot += float(model(x[i:i + mbs], y[i:i + mbs]))
    return tot / n


def run(c):
    if c['threads']:
        torch.set_num_threads(int(c['threads']))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    tr, va = load_tokens(c)
    model = GPT(vocab_size=VOCAB, num_layers=int(c['layers']), model_dim=int(c['dim']))
    if dev == 'cpu':
        model = model.float()  # bf16 matmuls are ~25x slower than fp32 on a laptop CPU
    model = model.to(dev)
    init_(model)
    opts = build_optimizers(model, c)
    T, B, mbs, steps = int(c['seq']), int(c['seqs']), int(c['mbs']), int(c['steps'])
    bs = B * T
    curve, t0, pos = [], time.time(), 0
    for step in range(steps):
        buf = tr[pos:pos + bs + 1].to(dev)
        pos += bs
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        tl = 0.0
        for i in range(0, B, mbs):
            loss = model(x[i:i + mbs], y[i:i + mbs])
            loss.backward()
            tl += float(loss)
        prog = step / steps
        eta = 1.0 if prog < 1 - c['cooldown'] else (1 - prog) / c['cooldown']
        for o in opts:
            for g in o.param_groups:
                g['lr'] = g['initial_lr'] * eta
            o.step()
        model.zero_grad(set_to_none=True)
        tl /= bs
        if step % 10 == 0 or step == steps - 1:
            curve.append([step, round(tl, 4)])
        if not np.isfinite(tl):
            break
    val = evaluate(model, va.to(dev), c) if np.isfinite(tl) else float('nan')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # MB on macOS (bytes)
    rec = dict(benchmark='NANOGPT-OPT-' + c['preset'], config=c, val=round(val, 4), train_last=round(tl, 4),
               curve=curve, secs=round(time.time() - t0, 1), peak_mb=round(rss), tokens=steps * bs)
    return rec


if __name__ == '__main__':
    c = parse(sys.argv[1:])
    rec = run(c)
    line = json.dumps(rec)
    print(line)
    if c['out']:
        with open(os.path.join(ROOT, c['out']) if not os.path.isabs(c['out']) else c['out'], 'a') as f:
            f.write(line + '\n')
