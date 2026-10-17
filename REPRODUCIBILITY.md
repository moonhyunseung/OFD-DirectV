# Reproducibility

Every empirical result must identify:

- repository commit SHA;
- experiment ID;
- exact command/config;
- hardware model and count;
- driver/CUDA/PyTorch/Triton versions as applicable;
- model/checkpoint revision when used;
- dtype and accumulation policy;
- warmup/repetition protocol;
- raw output path;
- analysis script SHA/path.

No result may be promoted to a claim from an unversioned notebook, copied terminal snippet, or profiler screenshot alone.

## Result layout

```text
results/<experiment_id>/<timestamp>/
  manifest.json
  raw/
  derived/
  summary.md
```

Negative and failed runs are retained when they constrain the claim boundary.
