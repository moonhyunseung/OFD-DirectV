# Results

Only manifest-backed results belong here.

Recommended structure:

```text
results/<experiment_id>/<timestamp>/
  manifest.json
  raw/
  derived/
  summary.md
```

`summary.md` must distinguish:

- directly measured values;
- derived quantities;
- interpretation/hypotheses;
- claim status affected by the result.

Negative, failed, or crossover-defining runs are retained when they constrain a claim.