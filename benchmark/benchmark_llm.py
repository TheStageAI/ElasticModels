import argparse
import json
import logging
import os
import timeit
from pathlib import Path

import torch
from transformers.pipelines import pipeline
from transformers import (
    AutoModelForCausalLM as HFAutoModelForCausalLM,
    AutoModelForImageTextToText as HFAutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.utils import logging as hf_logging

from gpu_monitor import GPUMemoryMonitor
from elastic_models.transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------
# Load model
# ----------------


def get_generator(args):
    if args.model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
        return get_vlm_generator(args)
    else:
        return get_llm_generator(args)


def get_vlm_generator(args):
    processor = AutoProcessor.from_pretrained(
        args.model_name, token=args.hf_token, padding_side="left"
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if args.mode == "original":
        model = HFAutoModelForImageTextToText.from_pretrained(
            args.model_name,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            mode=args.mode,
            device_map=args.device
        )
    model.eval()

    generator = pipeline(
        task="image-text-to-text", model=model, processor=processor, device=args.device
    )
    return generator


def get_llm_generator(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=args.hf_token, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "original":
        model = HFAutoModelForCausalLM.from_pretrained(
            args.model_name,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            mode=args.mode,
            device_map=args.device
        )
    model.eval()

    generator = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, device=args.device
    )
    return generator


def get_prompt(args):
    if args.prompt:
        prompt = args.prompt
    else:
        prompt_file = Path(__file__).parent / "prompts_llm.json"
        with open(prompt_file, "r") as f:
            data = json.load(f)
        prompt = data[args.input_context]

    prompt = [prompt] * args.batch_size
    return prompt


# -----------------
# Benchmarking
# -----------------


def run_cuda_synchronize(func):
    def wrapper():
        output = func()
        torch.cuda.synchronize()
        return output

    return wrapper


def time_generation(generator, prompt, number=1, repeat=5, **generate_kwargs):
    runs = timeit.repeat(
        run_cuda_synchronize(lambda: generator(prompt, **generate_kwargs)),
        number=number,
        repeat=repeat,
        setup="import torch; torch.cuda.synchronize()",
    )
    return min(runs) / number


def benchmark_time(
    generator, prompt, number=1, repeat=5, new_tokens=100, **generate_kwargs
):
    first = time_generation(
        generator,
        prompt,
        number=number,
        repeat=repeat,
        max_new_tokens=1,
        **generate_kwargs,
    )

    fastest = time_generation(
        generator,
        prompt,
        number=number,
        repeat=repeat,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        **generate_kwargs,
    )

    tps = (new_tokens - 1) / (fastest - first)
    return tps, first


def benchmark(
    generator,
    prompt,
    new_tokens=100,
    number=1,
    repeat=5,
    include_memory=True,
    **generate_kwargs,
):
    if isinstance(prompt, list) and "batch_size" not in generate_kwargs:
        generate_kwargs["batch_size"] = len(prompt)

    with GPUMemoryMonitor() as gpu_monitor:
        if not include_memory:
            gpu_monitor.stop()
        t, f = benchmark_time(
            generator,
            prompt,
            number=number,
            repeat=repeat,
            new_tokens=new_tokens,
            **generate_kwargs,
        )
    max_memory_usage = gpu_monitor.get_max_memory_usage()
    inputs = generator.tokenizer(prompt[0], return_tensors="pt")

    real_input_length = inputs["input_ids"].shape[1]

    results = {
        "tps": t,
        "ttft": f,
        "batch_size": len(prompt),
        "input_tokens": real_input_length,
    }
    if include_memory:
        results["max_memory_usage"] = max_memory_usage

    return results


def benchmark_quality(generator, args):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    model = generator.model
    tokenizer = generator.tokenizer

    # past_key_values = get_kvcache(model, args, max_cache_len)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device=args.device,
        dtype=torch.bfloat16,
        batch_size=args.batch_size,
    )
    out = {}
    for task_name in args.bench_tasks:
        out[task_name] = evaluator.simple_evaluate(
            model=lm,
            tasks=[task_name],
            # gen_kwargs={"past_key_values": past_key_values},
        )["results"]

    print(out)
    return out


# -----------------
# Check output
# -----------------


def check_output(generator, prompt, **generate_kwargs):
    if not isinstance(prompt, list):
        prompt = [prompt]

    if "batch_size" not in generate_kwargs:
        generate_kwargs["batch_size"] = len(prompt)

    outputs = generator(prompt, **generate_kwargs)
    samples = []
    for idx, (input, output) in enumerate(zip(prompt, outputs), start=1):
        print(f"\n# Input {idx}:\n{input}")
        output = output[0]["generated_text"].strip()
        print(f"\n# Output {idx}:\n{output}")
        samples.append(output)
    return samples


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Benchmark text generation for Elastic Models"
    )
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument(
        "--mode", default="original", choices=["original", "S", "M", "L", "XL"]
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""), type=str)
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Directory to store cache files",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--input_context", default="small", choices=["small", "medium", "large"]
    )
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--new_tokens", default=100, type=int)
    parser.add_argument("--check_output", action="store_true")

    parser.add_argument(
        "--bench_tasks",
        nargs="+",
        default=[],
        help="Tasks for quality benchmarks, e.g. `piqa`, `arc_challenge`, `mmlu`",
    )

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no_memory", action="store_true")
    args = parser.parse_args(args)
    args.device = torch.device(args.device)

    if not args.hf_token:
        logger.warning("HF_TOKEN environment variable is not set")

    return args


if __name__ == "__main__":
    args = parse_args()

    generator = get_generator(args)
    generate_kwargs = {}
    if args.mode != "original":
        generate_kwargs["cache_implementation"] = "paged"

    prompt = get_prompt(args)

    results = benchmark(
        generator,
        prompt,
        args.new_tokens,
        **generate_kwargs,
        include_memory=not args.no_memory,
    )
    print(f"Results for {args.mode} mode:")
    print(results)

    if args.bench_tasks:
        benchmark_quality(generator, args)

    if args.check_output:
        check_output(
            generator,
            prompt[-1],
            max_new_tokens=100,
            return_full_text=False,
            **generate_kwargs,
        )
