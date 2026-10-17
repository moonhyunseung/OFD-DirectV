# Tests

Required test layers:

1. representation/codec consistency;
2. exact identity tests against decode-then-aggregate reference;
3. signed-vs-offset representation equivalence;
4. floating-point numerical-contract tests;
5. edge cases for zero-length/short active ranges where applicable;
6. streaming composition tests once online-softmax integration is introduced;
7. optimized-kernel regression tests against the frozen reference.

Correctness tests must precede performance experiments.