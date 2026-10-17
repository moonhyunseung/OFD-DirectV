# H200: graph-amortized token-reduction tile boundary

## Frozen source and scope

- Graph source: df89f04c4c851a04ecc6525c60901f9b1b7546c5, experiments/h200_graph_boundary.py; SHA256 174ff55f19888523c8bdd671f601dda1ebda5b24ca1a01772fdd1ef4f690b646.
- Tile selection/evaluation: 641287013ba072a7c2bc2da5520a865ca095c9e1, experiments/h200_tile_holdout.py; SHA256 764914d779589895fea971860d110181b605004383dd49eacd973e81ab7c75cf.
- H200; torch 2.14.1+cu130; Triton 3.8.0. B=1, KV heads=1, r=6, d=256, group size=32.
- FP32 probabilities, unchanged FP16 scales, FP32 reductions/output. No probability downcast, quantizer change, token dropping or low-rank approximation.
- Resident CUDA Graph timing: eight warmups, 31 samples, 32 operations per replay, randomized method order. This does not measure serving or cold-HBM throughput.

## Selection without holdout tuning

Candidates were token tiles 64/128/256/512, each with offset and inline-signed code consumption. Training used lengths 1024/8192/131072 and seed 1731. One global tile/variant was selected by geometric mean relative time, then frozen for lengths 2048/16384/65536/262144 and seeds 1741/1742/1743.

Selected policy: **128 tokens, offset correction**, identical to the original default. This is a negative optimization result; no improvement over the default is claimed.

| Tile | Offset relative train time | Signed relative train time |
|---:|---:|---:|
| 64 | 1.979312 | 2.128540 |
| 128 | 1.000000 | 1.291669 |
| 256 | 1.010245 | 1.305853 |
| 512 | 1.015902 | 1.320732 |

All 96 declared numerical diagnostic cases passed before timing. These budgets include an absolute floor and are not a closed production numerical guarantee. The original 128 offset/signed compiled paths had 48/45 registers, zero spills, 512 shared bytes and no Tensor Core instruction.

## Held-out observation

Microseconds below are medians of three per-seed medians, rounded. The selected and default-offset functions are the same algorithm and were measured separately; their small differences are measurement variability.

| Length | Selected offset/128 | Default signed/128 | FP32 predecoded oracle |
|---:|---:|---:|---:|
| 2048 | 5.783 | 6.568 | 6.887 |
| 16384 | 16.818 | 23.159 | 15.879 |
| 65536 | 57.265 | 82.227 | 44.934 |
| 262144 | 204.186 | 301.071 | 151.457 |

Signed code consumption is slower than offset correction in these held-out measurements. The selected V path is slower than the predecoded FP32 oracle on the longer shapes. These results do not support transferring the K-side speed conclusion to V.

The operator still materializes partial [ceil(L/tile),r,d] outputs and reduces them in a separate PyTorch operation. The oracle excludes reconstruction and holds extra dense FP32 storage; neither baseline is an optimized fused quantized-serving engine.

## Decision and remaining gate

Retain offset/128 for the staged integration. Changes to token reduction need a new mechanism rather than an unsupported claim that a larger tile or signed representation is always better. Do not cast probabilities to FP16 merely to access Tensor Cores without a separately justified numerical contract. Full attention fusion, dynamic layouts, real-model numerical fidelity and optimized-serving comparisons remain separate gates.
