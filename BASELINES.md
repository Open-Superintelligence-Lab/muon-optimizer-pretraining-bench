# Baselines

Baselines for the main benchmark, NANOGPT-OPT: four current-best matrix optimizers from their official code, plus one Adam reference row.
Each one is used unchanged, pinned to a commit, and its license is in `baselines/official/`. Every field is in [baselines/manifest.json](baselines/manifest.json).

## Results

Laptop NANOGPT-OPT (4 layers x 128, 200 steps x 1,024 tokens, val loss on 32,768 FineWeb tokens, lower is better) against the published full-size scores (track 3, steps to val loss 3.28 with the 124M GPT, lower is better). Runs: `experiments/runs/NANOGPT-OPT.jsonl`, 14 runs, 7-13 minutes and 1.0-1.2 GB each on a loaded MacBook CPU.

| Method | lr x0.5 | official lr | lr x2 | Our best val | Published (track 3) | Beats Adam here | Beats Adam published |
|---|---|---|---|---|---|---|---|
| Moonlight | **7.4487** (.002) | 7.4598 (.004) | 7.4996 (.008) | 7.4487 | no entry; the paper reports about 2x compute efficiency over Adam | yes | yes (paper) |
| Muon | **7.4549** (.0125) | 7.4644 (.025) | 7.4895 (.05) | 7.4549 | 3250 steps (#36) | yes | yes |
| NorMuon | **7.4621** (.0175) | 7.4668 (.035) | 7.4752 (.07) | 7.4621 | 3250 steps (#10) | yes | yes |
| Adam (reference) | **7.5064** (.00075) | 7.6566 (.0015) | not run | 7.5064 | 5625 steps (#2, "most likely undertuned") | - | - |
| SOAP | **7.6345** (.0015) | 7.7430 (.003) | 7.8809 (.006) | 7.6345 | 3125 steps (#27, SOAP with a hyperball constraint, refresh every step) | no | yes |

- The published order (SOAP variant < NorMuon = Muon < Adam) holds for the three Muon-family methods: all beat Adam here by .044-.058. It fails for SOAP, which is last here and first on track 3. Two differences may explain it: track 3's SOAP adds a hyperball constraint and refreshes its basis every step (ours every 10, as in the SOAP README), and 200 steps give a basis refreshed only 20 times.
- Every method's best rate is the lowest rate tried, so no sweep brackets its best rate. The planned runs at lower rates (SOAP .00075, and Adam .003 for symmetry) are listed in `experiments/nanogpt_baselines.py` and not run yet.
- At 200 steps of 1,024 tokens the model is still close to the unigram stage (val about 7.45 nats from 10.83 at the start), so gaps under .01 (Moonlight, Muon, NorMuon) do not separate those three. `preset=full`, or more laptop steps, is needed for that.

## Sources and match checks

| Method | Official code | Commit | License | Match check |
|---|---|---|---|---|
| Muon | [KellerJordan/Muon](https://github.com/KellerJordan/Muon) `muon.py` (`SingleDeviceMuon`) | `f98f1ca` | MIT | passed |
| NorMuon | [zichongli5/NorMuon](https://github.com/zichongli5/NorMuon) `normuon.py` (`SingleDeviceNorMuon`) | `c6989a8` | MIT | passed |
| Moonlight | [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) `examples/toy_train.py` lines 49-240 (`Muon`) | `c2ad5b2` | MIT | passed |
| SOAP | [nikhilvyas/SOAP](https://github.com/nikhilvyas/SOAP) `soap.py` (`SOAP`) | `a1e5535` | MIT | passed |
| Adam | `torch.optim.Adam` | PyTorch | BSD-3-Clause | built-in, used as is |
| GPT model | [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) `records/track_3_optimization/train_gpt_simple.py` lines 58-158 | `4ea6b93` | MIT | passed |

`tests/check_official.py` runs the match checks after `sh third_party/fetch.sh` clones each repo at its commit:

1. Text: each file in `baselines/official/` and `env/nanogpt_arch.py` equals the upstream file, or the stated line range, byte for byte.
2. Model: the upstream GPT class and ours give the same loss on the same weights and input (1376.654419 both, 2 layers x 128, fp32).
3. Updates: for each optimizer, 4 steps on 128x512 and 512x128 matrices with the same gradients give exactly the upstream weights (max difference 0.00e+00).

A match check against a published number is not possible on a laptop: the published scores are steps to loss 3.28 with the 124M GPT on 8 GPUs. The checks above show the code is the official code; the order check below compares our laptop ranking with the published one.

## Changes from the official setups

- All methods: the matrix optimizer gets only the hidden matrices of the blocks. The embedding, the head and the 1-D parameters use the track-3 aux Adam groups (rates .7 / .004 / .015, betas (.8, .95), eps 1e-10, decoupled weight decay .001), the stable-then-linear-decay schedule with cooldown .7, and the official init, exactly as `train_gpt_simple.py`.
- All methods: laptop size (4 layers, width 128, context 256, 4 sequences per step, 200 steps, 204,800 training tokens, 32,768 validation tokens), the model in fp32 on CPU instead of bf16 on GPU (bf16 matmuls are about 25x slower on a laptop CPU). `preset=full` restores the official size.
- Muon: the Muon repo's Newton-Schulz with 5 steps, not track 3's own 12-step variant; the step count is fixed in this environment.
- Moonlight: its internal Adam parameter list left empty, since the aux groups already cover them; weight decay .1 and the update scale 0.2 sqrt(max(rows, cols)) as upstream.
- SOAP: betas (.95, .95), weight decay .01, preconditioner refresh every 10 steps, as in the SOAP README. Track 3's SOAP entries add a hyperball constraint and refresh every step; plain SOAP has no track-3 entry.
- Adam: the track-3 reference settings (lr .0015, betas (.9, .95), eps 1e-10, no weight decay on matrices).

Rates: each method ran at its official rate and at x0.5 and x2 (Adam: x1 and x0.5; Adam x2 and SOAP x0.25 are planned, not run). Weight decay stays at the official value.

## Why these four

- Muon is the method this environment studies and the track-3 reference (result #36, 3250 steps).
- NorMuon is the strongest published Muon variant at 1.1B and 5.4B parameters and ties tuned Muon on track 3 (result #10, 3250 steps).
- Moonlight's scaling and decoupled weight decay are the form of Muon used at the largest published scale (a 16B-parameter mixture of experts on 5.7T tokens, [arXiv 2502.16982](https://arxiv.org/abs/2502.16982)).
- SOAP is the strongest non-Muon matrix method in both recent benchmark papers ([arXiv 2509.02046](https://arxiv.org/abs/2509.02046), [arXiv 2509.01440](https://arxiv.org/abs/2509.01440)) and a SOAP variant holds the track-3 record (result #27, 3125 steps).

## Other strong recent methods (not included)

AdaMuon ([arXiv 2507.11005](https://arxiv.org/abs/2507.11005)), Dion and Dion2 ([arXiv 2504.05295](https://arxiv.org/abs/2504.05295), [arXiv 2512.16928](https://arxiv.org/abs/2512.16928)), MARS ([arXiv 2411.10438](https://arxiv.org/abs/2411.10438)), Scion ([arXiv 2502.07529](https://arxiv.org/abs/2502.07529)), SPlus ([arXiv 2506.07254](https://arxiv.org/abs/2506.07254)), Conda ([arXiv 2509.24218](https://arxiv.org/abs/2509.24218)) and Shampoo (track 3 result #21, 4100 steps).
