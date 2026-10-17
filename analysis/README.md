# Analysis

Analysis consumes frozen raw measurements and must preserve the distinction between theorem-level results and hardware observations.

Required practice:

- derive tables/figures reproducibly from versioned raw outputs;
- report context/batch/head crossover regions, including losses;
- distinguish token-reduction cost, representation work, and machine-lowering effects;
- never use profiler attribution to prove algebraic correctness;
- tie every durable conclusion to an experiment manifest and commit SHA.