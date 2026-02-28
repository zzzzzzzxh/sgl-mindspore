#!/usr/bin/env python3
"""
Run EAGLE3 draft model as standalone model on MindSpore backend.

This allows direct comparison of EAGLE3 draft model between different backends
without the overhead of speculative decoding.
"""


import sglang as sgl


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to EAGLE3 draft model"
    )
    parser.add_argument(
        "--device", type=str, default="npu", help="Device to run on (npu)"
    )
    parser.add_argument(
        "--model-impl", type=str, default="mindspore", help="Model implementation"
    )
    parser.add_argument(
        "--attention-backend", type=str, default="ascend", help="Attention backend"
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--port", type=int, default=24678, help="Server port")
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA graph (use for standalone mode)",
    )
    args = parser.parse_args()

    engine_args = {
        "model_path": args.model_path,
        "device": args.device,
        "model_impl": args.model_impl,
        "attention_backend": args.attention_backend,
        "tp_size": args.tp_size,
        "port": args.port,
        "log_level": "INFO",
        "mem_fraction_static": 0.75,
    }

    if args.disable_cuda_graph:
        engine_args["disable_cuda_graph"] = True

    print("Starting EAGLE3 standalone model server on MindSpore...")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Model implementation: {args.model_impl}")
    print(f"Attention backend: {args.attention_backend}")
    print(f"Disable CUDA graph: {args.disable_cuda_graph}")

    llm = sgl.Engine(**engine_args)

    prompts = [
        "what is speculative decoding?",
    ]

    # Use max_new_tokens instead of max_tokens for SGLang
    sampling_params = {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 100}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()
