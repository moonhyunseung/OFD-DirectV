# Manifests

Every durable DirectV experiment records:

- repository commit SHA;
- experiment ID and command/config;
- context length, batch, head/GQA geometry, group size;
- dtype and accumulation policy;
- hardware/toolchain;
- model/checkpoint revision if applicable;
- raw result paths;
- analysis script/path;
- signed-vs-offset representation choice.

Negative and failed runs remain part of the evidence record when they constrain a claim boundary.