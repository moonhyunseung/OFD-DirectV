# Reference implementations

This directory is for correctness references, not optimized kernels.

Planned files:

- `quant.py` — frozen groupwise INT4 codec contract;
- `decode_then_aggregate.py` — baseline decoded-quantized V path;
- `direct_offset.py` — exact offset-binary DirectV form;
- `direct_signed.py` — exact signed-code DirectV form;
- `streaming_reference.py` — later, exact online-softmax composition reference.

Reference code may unpack logical codes for clarity. It must not be presented as a performance implementation.