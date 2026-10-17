# Kernel gate

No optimized kernel belongs here before the algebraic and numerical contracts are frozen.

A kernel candidate must state:

- exact source and target representation;
- whether it uses offset-binary or signed-recoded codes;
- accumulation dtype;
- whether dense V is ever reconstructed internally;
- intended machine primitive/lowering;
- expected symbolic work/intermediate reduction;
- failure/crossover hypothesis.

Any kernel that reconstructs dense V is a baseline or diagnostic, not a DirectV implementation.