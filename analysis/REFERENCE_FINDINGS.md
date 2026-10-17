# Reference evidence — OFD-DirectV

- Source commit: `a02307c0bcc0bf5f8942b72ab8a9b0ecf4e4fe49`.
- Recorded run: `results/20261017T045216Z_239494f7`.
- Formal tests: 12 passed; see run.log. The source was committed and clean before execution.
- No GPU or performance measurement was performed. No original OFD E00–E14 result was modified.

## Direct observations

| Compute | Rows | Max absolute difference | Max defined relative L2 | Zero-reference/nonzero-error rows |
|---|---:|---:|---:|---:|
| torch.float64 | 48 | 1.03318598121e-09 | 2.50387258812e-14 | 3 |
| torch.float32 | 48 | 0.25390625 | 5.0214830494e-06 | 3 |

Largest absolute difference: case `max_half_scale`, L=129, `torch.float32`, `offset`. Raw effects are preserved; no thresholds were tuned to them.

All outputs in this profile were finite. Relative L2 is explicitly undefined when the baseline is exactly zero but reassociation produces a nonzero residual; these rows are neither removed nor reported as zero error. Maxima are across different fixtures, not simultaneous guarantees.
- All-code8 zero-state residual, torch.float64: max_abs=4.4408920985e-15.
- All-code8 zero-state residual, torch.float32: max_abs=1.66893005371e-06.

## Interpretation

Offset correction can suffer cancellation; the signed byte path removes that algebraic correction. The same real operator does not imply bitwise equality. The FP16 scale-extreme fixtures explain why absolute and normalized error must be shown together. Stored zero scales are characterized as a degenerate extension to the positive-scale contract, preserving the original quantizer.

## Not established

No production accumulation policy or numerical budget is closed. CPU references materialize logical codes (and V coefficients); they do not verify a native packed GPU path. No measured speedup, HBM attribution, crossover, or model quality claim follows.

## Next justified work

Freeze the intended production accumulation/cast/exp policy and its numerical acceptance budget before native lowering. Inspect generated code for hidden reconstruction, then run matched hardware boundary measurements. Do not change packing/quantizer or add approximations to obtain a favorable result.
