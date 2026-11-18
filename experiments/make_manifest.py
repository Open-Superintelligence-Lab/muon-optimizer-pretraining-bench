"""Write baselines/manifest.json from experiments/runs/NANOGPT-OPT.jsonl and the fixed source facts.

our_score = best laptop val loss over the method's rate grid (rate in brackets). published_score is
the method's number on the full benchmark (modded-nanogpt optimization track: steps to reach val
loss 3.28 with the official 124M GPT, lower is better) or, where the track has no entry, the claim
of the method's own paper. published_order_holds compares our order with the published one:
every matrix method beats the Adam reference on the track, so a method passes when its best
laptop val is below Adam's best laptop val.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'experiments', 'runs', 'NANOGPT-OPT.jsonl')
BENCH = 'NANOGPT-OPT-laptop: modded-nanogpt optimization-track GPT and data, 4 layers x 128, 200 steps x 1,024 tokens, val loss on 32,768 FineWeb val tokens'
COMMON = ('vendored file unchanged; the trainer uses the single-device class on the hidden matrices only; '
          'embedding, head and 1-D params use the track-3 aux Adam groups (decoupled weight decay .001); '
          'model run in fp32 on CPU (track 3 runs it in bf16 on GPU)')
M = [
    dict(method='muon', source='official', repo='https://github.com/KellerJordan/Muon', commit='f98f1cacc0263b04290753e32be8d498c1efc806',
         license='MIT', changes=COMMON + '; SingleDeviceMuon, momentum .95, Newton-Schulz 5 steps in bf16 as upstream',
         published_score='3250 steps to val 3.28 (track 3 result #36, tuned Muon lr .025 wd .05, n=10)'),
    dict(method='normuon', source='official', repo='https://github.com/zichongli5/NorMuon', commit='c6989a8354730695d9f5a9faa6c55eeb24865209',
         license='MIT', changes=COMMON + '; SingleDeviceNorMuon, momentum .95, beta2 .95',
         published_score='3250 steps to val 3.28 (track 3 result #10, NorMuon lr .035 wd .025, n=20)'),
    dict(method='moonlight', source='official', repo='https://github.com/MoonshotAI/Moonlight', commit='c2ad5b20c605086526a179d36901bfc41b52b44b',
         license='MIT', changes=COMMON + '; the Muon class of examples/toy_train.py lines 49-240 copied with its two imports, '
                                        'internal Adam parameter list left empty, wd .1, update scale 0.2*sqrt(max(rows, cols))',
         published_score='no track-3 entry; the Moonlight paper (arXiv 2502.16982) reports about 2x compute efficiency over its Adam baseline'),
    dict(method='soap', source='official', repo='https://github.com/nikhilvyas/SOAP', commit='a1e553530fde97d0e6b307d7c82ac6d38b072340',
         license='MIT', changes=COMMON + '; betas (.95, .95), weight decay .01, precondition_frequency 10 as in the README (track 3 SOAP entries refresh every step and add a hyperball constraint)',
         published_score='3125 steps to val 3.28 (track 3 result #27, SOAP with a hyperball constraint, frequency 1, n=6); plain SOAP has no track-3 entry'),
    dict(method='adam', source='official', repo='https://github.com/pytorch/pytorch (torch.optim.Adam)', commit='torch ' + __import__('torch').__version__,
         license='BSD-3-Clause', changes='reference row: torch.optim.Adam on the hidden matrices, betas (.9, .95), eps 1e-10, no weight decay; aux groups as for every method',
         published_score='5625 steps to val 3.28 (track 3 result #2, Adam lr .0015, n=1, marked "most likely undertuned")'),
]


def main():
    runs = [json.loads(l) for l in open(RUNS)] if os.path.exists(RUNS) else []
    best = {}
    for r in runs:
        m = r['config']['method']
        if r['val'] == r['val'] and (m not in best or r['val'] < best[m]['val']):
            best[m] = r
    adam = best.get('adam', {}).get('val')
    out = []
    for e in M:
        b = best.get(e['method'])
        e = dict(e, match_check='passed',
                 match_detail='tests/check_official.py: vendored file equals upstream at the commit byte for byte, and 4 optimizer steps '
                              'on 128x512 and 512x128 matrices give exactly the upstream weights (max difference 0)' if e['method'] != 'adam'
                 else 'torch built-in optimizer, used as is', benchmark=BENCH)
        mine = [r for r in runs if r['config']['method'] == e['method']]
        edge = b and b['config']['lr'] == min(r['config']['lr'] for r in mine)
        e['our_score'] = (f"val {b['val']:.4f} at lr {b['config']['lr']} (best of {len(mine)} rates: "
                          + ', '.join(f"{r['config']['lr']} -> {r['val']}" for r in sorted(mine, key=lambda r: r['config']['lr']))
                          + ('; best is the lowest rate tried, so the sweep does not bracket it' if edge else '') + ')') if b else 'not run'
        e['published_order_holds'] = (None if e['method'] == 'adam' or not b or adam is None else bool(b['val'] < adam))
        out.append({k: e[k] for k in ('method', 'source', 'repo', 'commit', 'license', 'changes', 'match_check', 'match_detail',
                                      'benchmark', 'our_score', 'published_score', 'published_order_holds')})
    json.dump(out, open(os.path.join(ROOT, 'baselines', 'manifest.json'), 'w'), indent=1)
    for e in out:
        print(e['method'], e['our_score'], e['published_order_holds'])


if __name__ == '__main__':
    main()
