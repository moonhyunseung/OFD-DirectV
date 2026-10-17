# Numerical contract

## Exactness levels

`ALGEBRAIC-EXACT` means the direct and decode-first forms define the same real-valued decoded-quantized operator.

`BITWISE-EXACT` would require identical IEEE floating-point evaluation order and is not assumed by default.

## Reference comparison

The numerical baseline is the frozen decoded quantized V, not the pre-quantized FP V.

Comparison paths:

```text
Path A: packed V -> decode -> P @ Vhat
Path B: packed V -> coefficient prescale -> code reduction -> broadcast correction
```

Both paths must use explicitly declared:

- input dtype;
- stored scale dtype;
- accumulation dtype;
- cast points;
- reduction order where relevant.

## Acceptance

Tolerances must be established from the intended production accumulation policy, not chosen to make a kernel pass. FP64/FP32 references may be used to separate algebraic differences from rounding-order differences.

## Forbidden interpretation

A small numerical difference caused solely by reassociation does not convert an algebraically exact rewrite into a model approximation. Conversely, numerical closeness cannot prove the algebraic identity.