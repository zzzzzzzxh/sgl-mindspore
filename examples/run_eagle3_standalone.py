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
    args = parser.parse_args()

    print("Starting EAGLE3 standalone model server on MindSpore...")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Model implementation: {args.model_impl}")
    print(f"Attention backend: {args.attention_backend}")

    llm = sgl.Engine(
        model_path=args.model_path,
        device=args.device,
        model_impl=args.model_impl,
        attention_backend=args.attention_backend,
        tp_size=args.tp_size,
        port=args.port,
        log_level="INFO",
        mem_fraction_static=0.75,
    )

    prompts = [
        "what is speculative decoding?",
        "explain machine learning",
    ]

    sampling_params = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 100}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()
