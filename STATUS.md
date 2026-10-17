# Status

## Current phase

**T1 — Reference verified; production numerical contract open**

No hardware or speed claim exists yet.

| Stage | Requirement | Status |
|---|---|---|
| T0 Theory | Exact V-side operator and packed representation contract written | PASS |
| T1 Reference | Decode-first and direct references | PASS (CPU reference) |
| T2 Numerical | Floating-point contract fixed | IN PROGRESS (CPU profiles fixed; production policy open) |
| T3 Lowering | Packed token reduction mapped to machine primitives | NOT STARTED |
| T4 Kernel | Native microkernel | NOT STARTED |
| T5 Boundary | Failure/crossover region | NOT STARTED |
| T6 Model | Real decode integration | NOT STARTED |
| T7 Freeze | Claims/results/manifests frozen | NOT STARTED |

## Original bootstrap questions (see checkpoint below for resolved items)

1. Prove and formalize coefficient prescaling `W=P*S_V` as the canonical exact V form.
2. Compare offset correction vs signed-nibble recoding.
3. Determine the minimal streaming state required when DirectV is composed with online softmax.
4. Only after these are closed, inspect hardware lowering.

## Verified CPU reference checkpoint — 2026-10-17

Source `a02307c0bcc0bf5f8942b72ab8a9b0ecf4e4fe49`; [12-test formal run and raw evidence](results/20261017T045216Z_239494f7/FINDINGS.md). Exact dependencies are pinned where used. These are CPU reference results only.
