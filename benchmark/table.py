"""Print the NANOGPT-OPT baseline table from experiments/runs/NANOGPT-OPT.jsonl.

python3 benchmark/table.py      # best laptop val loss per optimizer (lower is better), with every rate tried
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = [json.loads(l) for l in open(os.path.join(ROOT, 'experiments', 'runs', 'NANOGPT-OPT.jsonl')) if l.strip()]
by = {}
for r in runs:
    by.setdefault(r['config']['method'], []).append(r)
rows = sorted(by.items(), key=lambda kv: min(r['val'] for r in kv[1]))
print('| Optimizer | Best val loss | Best lr | All rates tried (lr -> val) |')
print('|---|---|---|---|')
for m, rs in rows:
    b = min(rs, key=lambda r: r['val'])
    tried = ', '.join(f"{r['config']['lr']} -> {r['val']:.4f}" for r in sorted(rs, key=lambda r: r['config']['lr']))
    print(f"| {m} | **{b['val']:.4f}** | {b['config']['lr']} | {tried} |")
