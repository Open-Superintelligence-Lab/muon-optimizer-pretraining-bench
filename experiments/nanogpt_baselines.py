"""Run the NANOGPT-OPT baseline grid: each official method at its official/paper rate and at half
and twice that rate. Results go to experiments/runs/NANOGPT-OPT.jsonl; finished configs are skipped.

    python3 -I experiments/nanogpt_baselines.py --jobs 2 --ac-only --max-secs 300
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'experiments', 'runs', 'NANOGPT-OPT.jsonl')
BASE = {'muon': (0.025, 0.05), 'normuon': (0.035, 0.025), 'moonlight': (0.004, 0.1), 'soap': (0.003, 0.01), 'adam': (0.0015, 0.0)}
GRID = [dict(method=m, lr=round(lr * f, 6), wd=wd) for m, (lr, wd) in BASE.items() for f in (1, 0.5, 2)]
GRID.append(dict(method='soap', lr=0.00075, wd=0.01))  # SOAP was best at the lowest grid rate (.0015), so one lower rate brackets it
RUN_SECS = 420  # measured: about 7 minutes per laptop run with two in parallel under load


def done():
    if not os.path.exists(OUT):
        return set()
    return {(r['config']['method'], r['config']['lr'], r['config']['wd']) for r in map(json.loads, open(OUT))}


def on_battery():
    return 'AC Power' not in subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout


def main(argv):
    jobs = int(argv[argv.index('--jobs') + 1]) if '--jobs' in argv else 2
    max_secs = float(argv[argv.index('--max-secs') + 1]) if '--max-secs' in argv else 1e9
    t0 = time.time()
    todo = [g for g in GRID if (g['method'], g['lr'], g['wd']) not in done()]
    print(len(todo), 'runs to do')
    while todo:
        if '--ac-only' in argv and on_battery():
            print('stopped: on battery'); break
        if time.time() - t0 + RUN_SECS > max_secs:
            print('stopped: time budget'); break
        batch, todo = todo[:jobs], todo[jobs:]
        ps = [subprocess.Popen([sys.executable, '-I', os.path.join(ROOT, 'env', 'nanogpt_train.py'),
                                *[f'{k}={v}' for k, v in g.items()], 'out=' + OUT],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for g in batch]
        for p, g in zip(ps, batch):
            print(g, 'exit', p.wait())
    print('left:', len(todo))


if __name__ == '__main__':
    main(sys.argv[1:])
