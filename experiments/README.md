# Experiments

Experiments exist to locate the boundary of an already-defined exact DirectV form.

Every experiment must declare:

1. theorem/identity already established;
2. single unresolved implementation or hardware assumption;
3. exact failure condition;
4. boundary variables;
5. minimal measurement;
6. stopping rule;
7. why algebra alone cannot decide it.

Forbidden: exploratory sweeps used to decide algebra, approximate token dropping labeled exact, or performance claims before native lowering is inspected.

Planned series:

- E00 reference/numerical equivalence;
- E01 offset vs signed exact form;
- E02 token-reduction lowering;
- E03 context-length crossover;
- E04 batch/head crossover;
- E05 online-softmax composition;
- E06 model decode integration.
