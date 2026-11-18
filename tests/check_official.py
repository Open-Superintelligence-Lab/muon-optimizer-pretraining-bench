"""Match checks for the NANOGPT-OPT baselines against the official repos.

1. Text: each vendored file equals the upstream file (or line range) at the pinned commit, byte for byte.
2. Numbers: the upstream code, loaded straight from third_party/, and our copy give the same model
   loss and the same optimizer update on a small input (4 steps, width 128, 2 layers).

Needs `sh third_party/fetch.sh` first. Run with python3 -I (third_party/ is untrusted: its files are
only read as text here and executed in a fresh namespace, never imported from their own folder).
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'baselines', 'official'))
import torch  # noqa: E402

TP = os.path.join(ROOT, 'third_party')
OFF = os.path.join(ROOT, 'baselines', 'official')
MARK = '# ---- upstream lines'
CASES = [  # (ours, upstream, line range or None, header lines to skip in ours)
    ('baselines/official/muon.py', 'Muon/muon.py', None),
    ('baselines/official/normuon.py', 'NorMuon/normuon.py', None),
    ('baselines/official/soap.py', 'SOAP/soap.py', None),
    ('baselines/official/moonlight_muon.py', 'Moonlight/examples/toy_train.py', (49, 240)),
    ('env/nanogpt_arch.py', 'modded-nanogpt/records/track_3_optimization/train_gpt_simple.py', (58, 158)),
]


def upstream_text(rel, rng):
    lines = open(os.path.join(TP, rel)).read().splitlines(keepends=True)
    return ''.join(lines[rng[0] - 1:rng[1]]) if rng else ''.join(lines)


def ours_text(rel, rng):
    s = open(os.path.join(ROOT, rel)).read()
    if rng:
        s = s.split('\n', [i for i, l in enumerate(s.split('\n')) if l.startswith(MARK)][0] + 1)[-1]
    return s


def load_upstream(rel, rng, prelude):
    mod = types.ModuleType('up_' + os.path.basename(rel).replace('.', '_'))
    exec(compile(prelude + upstream_text(rel, rng), 'upstream:' + rel, 'exec'), mod.__dict__)
    return mod


def main():
    if not os.path.isdir(os.path.join(TP, 'Muon')):
        raise SystemExit('run `sh third_party/fetch.sh` first')
    bad = 0
    for ours, up, rng in CASES:
        same = ours_text(ours, rng) == upstream_text(up, rng)
        print(f"text  {ours:40s} == {up}{' L%d-%d' % rng if rng else ''}: {'passed' if same else 'FAILED'}")
        bad += not same

    pre = 'import math\nimport torch\nfrom torch import Tensor, nn\nimport torch.nn.functional as F\nimport torch.distributed as dist\n'
    # model: same loss from the upstream class and ours on the same weights
    from env.nanogpt_arch import GPT
    up_arch = load_upstream(CASES[4][1], CASES[4][2], pre)
    torch.manual_seed(0)
    a = GPT(50304, 2, 128).float()
    b = up_arch.GPT(50304, 2, 128).float()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 50257, (2, 64))
    xi, yi = x[:, :-1].contiguous(), x[:, 1:].contiguous()
    la, lb = float(a(xi, yi)), float(b(xi, yi))
    ok = la == lb
    print(f'model loss ours {la:.6f} upstream {lb:.6f}: {"passed" if ok else "FAILED"}')
    bad += not ok

    # optimizers: 4 steps on the same weights and gradients, ours vs upstream
    import muon, normuon, soap, moonlight_muon  # noqa: E401
    ups = {k: load_upstream(r, g, pre) for k, r, g in [('muon', 'Muon/muon.py', None), ('normuon', 'NorMuon/normuon.py', None),
                                                        ('soap', 'SOAP/soap.py', None), ('moon', 'Moonlight/examples/toy_train.py', (49, 240))]}
    mk = {
        'muon': lambda M, ps: M.SingleDeviceMuon(ps, lr=0.025, weight_decay=0.05),
        'normuon': lambda M, ps: M.SingleDeviceNorMuon(ps, lr=0.035, weight_decay=0.025),
        'soap': lambda M, ps: M.SOAP(ps, lr=0.003, betas=(0.95, 0.95), weight_decay=0.01, precondition_frequency=2),
        'moon': lambda M, ps: M.Muon(lr=0.004, wd=0.1, muon_params=ps, adamw_params=[]),
    }
    mods = {'muon': muon, 'normuon': normuon, 'soap': soap, 'moon': moonlight_muon}
    for k in mk:
        torch.manual_seed(1)
        w0 = [torch.randn(128, 512), torch.randn(512, 128)]
        gs = [[torch.randn_like(w) for w in w0] for _ in range(4)]
        outs = []
        for M in (mods[k], ups[k]):
            ps = [torch.nn.Parameter(w.clone()) for w in w0]
            opt = mk[k](M, ps)
            for g in gs:
                for p, gi in zip(ps, g):
                    p.grad = gi.clone()
                opt.step()
            outs.append([p.detach() for p in ps])
        d = max(float((u - v).abs().max()) for u, v in zip(*outs))
        ok = d <= 1e-6
        print(f'update {k:8s} max |ours - upstream| after 4 steps = {d:.2e}: {"passed" if ok else "FAILED"}')
        bad += not ok
    print('ALL PASSED' if not bad else f'{bad} FAILED')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
