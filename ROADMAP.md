# Roadmap

## Phase 1 — Algebra closure

- Freeze shapes, group semantics, and token-axis reduction.
- Prove the coefficient-prescaled direct form.
- Maintain both offset-binary and signed-recoded exact variants.
- Derive exact streaming composition requirements with online softmax.

Exit criterion: every V-side rewrite has a pointwise proof or explicit representation bijection.

## Phase 2 — Reference and numerical semantics

- Implement decode-then-aggregate and direct reference paths.
- Separate `ALGEBRAIC-EXACT` from bitwise floating-point identity.
- Fix accumulation dtypes and acceptance criteria before optimization.

## Phase 3 — Hardware lowering

- Determine packed-code consumption strategy along the token axis.
- Inspect unpack/recode, register reuse, reduction structure, and generated instructions.
- Reject any candidate that reconstructs dense V internally.

## Phase 4 — Boundary experiments

Measure only the failure/crossover boundary of an exact lowering. Primary variables: context length, batch, query-head ratio, accumulation dtype, tile geometry, and signed-vs-offset representation.

## Phase 5 — Model and online-softmax integration

- Validate standalone model decode.
- Prove and then implement composition with streaming softmax.
- Preserve negative results and measured crossover regions.

## Phase 6 — Freeze

Freeze source SHAs, model revision, hardware/toolchain, manifests, raw measurements, and final claim statuses.
