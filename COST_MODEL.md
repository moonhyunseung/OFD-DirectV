# DirectV symbolic cost model

This file counts logical work/intermediates only. It does not claim GPU latency.

Let

```text
r = query heads per KV head
L = active tokens
G = quantization groups
m = coordinates per group
d = G*m
```

## Decode-first path

A logical decode-first path materializes `L*G*m` decoded V values and applies the group scale to every coordinate before value aggregation.

The aggregation then performs the generic token reduction for every output coordinate.

## Direct path

Define

```text
W[h,t,g] = P[h,t] * S_V[t,g]
```

and compute

```text
Y_code[h,g,j] = sum_t W[h,t,g] * C_V[t,g,j]
B[h,g] = 8 * sum_t W[h,t,g]
Y[h,g,j] = Y_code[h,g,j] - B[h,g].
```

Logical consequences:

- dense `L*G*m` decoded-V materialization disappears;
- scale application moves from `L*G*m` decoded coordinates into `r*L*G` coefficients;
- zero-point correction becomes `r*G` scalar token reductions plus broadcast across `m` coordinates;
- the generic `r*L*G*m` token-coordinate interaction remains unless a separate exact factorization is proved.

## What is not implied

The symbolic reduction does not imply lower latency, fewer machine instructions, or better Tensor Core utilization. Unpack/recode, reduction scheduling, registers, shared memory, and occupancy belong to hardware lowering.

## Signed representation

Signed recoding removes the explicit zero-point term algebraically. Whether the recode itself is cheaper than offset correction is a separate machine-level boundary question.