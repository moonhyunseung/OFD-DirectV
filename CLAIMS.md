# Claims

| ID | Claim | Type | Status | Evidence required |
|---|---|---|---|---|
| V-C0 | The packed V representation is completely specified. | Representation | OPEN | Frozen codec contract |
| V-C1 | Dense dequantized V is not mathematically required for decoded-quantized value aggregation. | Algebraic | PROVED AT FORMULA LEVEL | Exact derivation |
| V-C2 | Per-group V scale can be absorbed into the probability coefficient `W=P*S_V` exactly. | Algebraic | PROVED AT FORMULA LEVEL | Distributivity |
| V-C3 | Offset correction reduces to a scalar `(h,g)` token reduction broadcast across width `m`. | Algebraic | PROVED AT FORMULA LEVEL | Exact identity |
| V-C4 | Signed-nibble recoding can remove explicit offset correction without changing the represented quantized integer. | Representation | OPEN | Exhaustive code-space verification |
| V-C5 | DirectV can be accumulated inside an exact online-softmax streaming state without materializing full P. | Composition | OPEN | Composition proof |
| V-C6 | A representation-native packed V lowering outperforms decode-first execution in at least one workload regime. | Hardware/empirical | OPEN | Verified lowering + boundary experiment |
| V-C7 | The gain transfers to real model decode under the declared numerical semantics. | System/empirical | OPEN | Model integration |

## Guardrails

- V-C1–V-C3 are statements about decoded-quantized semantics, not pre-quantized FP V.
- V-C5 must be proved compositionally before fusion experiments.
- No universal speedup claim is allowed.
