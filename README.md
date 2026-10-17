# OFD-DirectV

**Exact representation-native value aggregation from packed quantized V, without materializing dense dequantized values.**

This repository isolates the V-side operator from the broader OFD research line and studies it independently to theorem, reference, kernel, boundary, and model-integration closure.

## Scope

Frozen decoded quantized semantics:

```text
Vhat[t,g,j] = S_V[t,g] * (C_V[t,g,j] - 8)
```

Target operator:

```text
Y[h,g,j] = sum_t P[h,t] * Vhat[t,g,j]
```

Exact direct form:

```text
W[h,t,g] = P[h,t] * S_V[t,g]
Y[h,g,j]
= sum_t W[h,t,g] * C_V[t,g,j]
  - 8 * sum_t W[h,t,g]
```

The direct form removes dense V dequantization as a mathematical intermediate. The main reduction axis is the token dimension, so DirectV is not treated as a copy of DirectK.

## Research order

1. representation contract;
2. exact algebra;
3. reference implementation;
4. numerical contract;
5. token-reduction cost model;
6. hardware lowering;
7. boundary experiments;
8. model integration;
9. freeze.

## Non-goals

No truncated SVD, token dropping, block dropping, numerical low-rank approximation, or quality-changing quantization modification belongs in the exact path.

## Rule

Experiments do not test whether the identity is true. They test numerical realization or the hardware boundary of an already-proved exact form.

## Reproducible reference checkpoint

See [reference findings](analysis/REFERENCE_FINDINGS.md) for the pinned source SHA, raw run, numerical observations and remaining gates. Install the recorded environment and run `python experiments/verify_reference.py` from a clean worktree. No GPU performance claim is made.
