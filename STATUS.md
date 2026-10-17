# Status

## Current phase

**T0 — Theory / semantic contract**

No hardware or speed claim exists yet.

| Stage | Requirement | Status |
|---|---|---|
| T0 Theory | Exact V-side operator and packed representation contract written | PASS |
| T1 Reference | Decode-first and direct references | NOT STARTED |
| T2 Numerical | Floating-point contract fixed | NOT STARTED |
| T3 Lowering | Packed token reduction mapped to machine primitives | NOT STARTED |
| T4 Kernel | Native microkernel | NOT STARTED |
| T5 Boundary | Failure/crossover region | NOT STARTED |
| T6 Model | Real decode integration | NOT STARTED |
| T7 Freeze | Claims/results/manifests frozen | NOT STARTED |

## Immediate questions

1. Prove and formalize coefficient prescaling `W=P*S_V` as the canonical exact V form.
2. Compare offset correction vs signed-nibble recoding.
3. Determine the minimal streaming state required when DirectV is composed with online softmax.
4. Only after these are closed, inspect hardware lowering.
