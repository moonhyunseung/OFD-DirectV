# Exact DirectV identity

## Frozen decoded-quantized semantics

For token `t`, group `g`, coordinate `j`:

```text
Vhat[t,g,j] = S_V[t,g] * (C_V[t,g,j] - 8)
```

with group size `m` and `C_V` the stored unsigned nibble code.

The decoded value aggregation is

```text
Y[h,g,j] = sum_t P[h,t] * Vhat[t,g,j].
```

Substitution gives

```text
Y[h,g,j]
= sum_t P[h,t] * S_V[t,g] * (C_V[t,g,j] - 8).
```

Define

```text
W[h,t,g] = P[h,t] * S_V[t,g].
```

Then, by distributivity,

```text
Y[h,g,j]
= sum_t W[h,t,g] * C_V[t,g,j]
  - 8 * sum_t W[h,t,g].
```

The second term is independent of coordinate `j` inside group `g`, so it is a scalar correction per `(h,g)` broadcast across the group width.

## Guarantee

Let `D_V(C,S)` be the frozen decoder and `A(P,V)=PV` the value-aggregation operator. Define the direct operator `A_direct(P,C,S)` by the identity above. Over real arithmetic,

```text
A_direct(P,C,S) = A(P, D_V(C,S))
```

for every admissible `P,C,S` under the frozen groupwise representation contract.

No model-distribution assumption is required.

## Consequences

- dense decoded V is not a mathematically required intermediate;
- group scale moves from every decoded coordinate into the reduced coefficient `W`;
- offset correction becomes one scalar token reduction per `(h,g)`, then broadcast;
- the generic token-by-coordinate contraction remains and is not claimed to disappear.

## Signed normal form

If the offset-binary nibble is recoded exactly to signed integer `q=C-8`, then

```text
Y[h,g,j] = sum_t P[h,t] * S_V[t,g] * q[t,g,j].
```

This removes explicit offset correction but does not by itself establish a hardware speedup.

## Exactness level

The identities above are `ALGEBRAIC-EXACT`. Different floating-point evaluation orders can round differently; bitwise equality is a separate numerical question.