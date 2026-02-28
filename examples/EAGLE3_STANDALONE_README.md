# EAGLE3 Standalone Model for MindSpore

This directory for running EAGLE3 draft models as standalone models without speculative decoding on Ascend NPU.

## Usage

```bash
python examples/run_eagle3_standalone.py \
    --model-path /path/to/Llama-3-8B-Eagle3 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --tp-size 1
```

## Purpose

- Benchmark EAGLE3 draft model independently
- Compare performance with SGLang native backend
- Profile MindSpore-specific optimizations
