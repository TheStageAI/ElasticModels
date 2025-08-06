import argparse
import logging
import os
import random
import timeit
from pathlib import Path
from typing import List, Optional
from logger import _LOGGER_MAIN

import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoProcessor
from transformers.pipelines import pipeline
from transformers.models.musicgen.modeling_musicgen import (
    MusicgenForConditionalGeneration as HFMusicgenForConditionalGeneration,
)
from transformers.utils import logging as hf_logging

from gpu_monitor import GPUMemoryMonitor
from elastic_models.transformers import MusicgenForConditionalGeneration

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------
# Load model
# ----------------


def get_generator(args):
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )

    if args.mode == "original":
        model = HFMusicgenForConditionalGeneration.from_pretrained(
            args.model_name,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            torch_dtype=args.dtype,
        )
    else:
        model = MusicgenForConditionalGeneration.from_pretrained(
            args.model_name,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            torch_dtype=args.dtype,
            mode=args.mode,
            device_map=args.device
        )

    model.eval()

    synthesiser = pipeline(
        task="text-to-audio",
        model=model,
        tokenizer=processor.tokenizer,
        device=args.device,
    )

    return synthesiser


# -----------------
# Benchmarking
# -----------------
def run_cuda_synchronize(func):
    def wrapper():
        output = func()
        torch.cuda.synchronize()
        return output

    return wrapper


def trim_input(generator):
    counter = 0

    def wrapper(prompt, **kw):
        nonlocal counter
        counter += 1
        if counter % 2 == 0:
            new_prompt = []
            for p in prompt:
                new_prompt.append(" ".join(p.split(" ")[:-1]))
            return generator(new_prompt, **kw)
        else:
            return generator(prompt, **kw)

    return wrapper


def time_generation(generator, prompt, number=1, repeat=5, **generate_kwargs):
    trim_generator = trim_input(generator)
    batch_size = generate_kwargs.pop("batch_size", 1)
    runs = timeit.repeat(
        run_cuda_synchronize(
            lambda: trim_generator(
                prompt, batch_size=batch_size, generate_kwargs=generate_kwargs
            )
        ),
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

    results = {
        "tps": t,
        "ttft": f,
        "batch_size": len(prompt),
    }
    if include_memory:
        results["max_memory_usage"] = max_memory_usage

    return results


# -----------------
# Check output
# -----------------


def check_output(generator, prompt, args, folder, **generate_kwargs):
    if not isinstance(prompt, list):
        prompt = [prompt]

    set_seed(args.seed)
    outputs = generator(prompt, generate_kwargs=generate_kwargs)

    sampling_rate = outputs[0]["sampling_rate"]
    for idx, (input, output) in enumerate(zip(prompt, outputs), start=1):
        filename = f"{input.replace(' ', '_')}_{args.seed}_idx{idx}.wav"
        wavfile.write(folder / filename, rate=sampling_rate, data=output["audio"])


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark text generation for Elastic Models"
    )
    parser.add_argument("--model_name", default="facebook/musicgen-large")
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

    parser.add_argument("--cache", default="paged", choices=["flexi-static", "paged"])

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--prompt", type=str, default="Mozart's Symphony No. 40 in G minor, K. 550"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--new_tokens", default=256, type=int)
    parser.add_argument(
        "--output_dir", type=str, default="", help="Directory to save generated tracks"
    )

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default=torch.float16, help="dtype")
    parser.add_argument("--no_memory", action="store_true")
    args = parser.parse_args(args)
    args.device = torch.device(args.device)

    if not args.hf_token:
        logger.warning("HF_TOKEN environment variable is not set")

    if args.mode == "XXL":
        args.cache = "paged"

    return args


if __name__ == "__main__":
    args = parse_args()

    _LOGGER_MAIN.info(
        f"Loading model {args.model_name} in {args.mode} mode."
    )
    generator = get_generator(args)
    _LOGGER_MAIN.info(
        f"Model {args.model_name} in {args.mode} mode loaded successfully."
    )
    generate_kwargs = {"disable_compile": True}
    if args.mode != "original":
        generate_kwargs["cache_implementation"] = args.cache

    prompt = [args.prompt] * args.batch_size

    _LOGGER_MAIN.info(
        f"Starting latency benchmark for {args.model_name} in {args.mode} mode"
    )
    results = benchmark(
        generator,
        prompt,
        args.new_tokens,
        batch_size=args.batch_size,
        **generate_kwargs,
        include_memory=not args.no_memory,
    )
    _LOGGER_MAIN.info(f"Latency benchmark for {args.model_name} in {args.mode} are ready:")
    for key, value in results.items():
        _LOGGER_MAIN.info(f"{key}: {value}")
    _LOGGER_MAIN.info("Latency benchmarking completed.")
    # print(f"Results for {args.mode} mode:")
    # print(results)

    if args.output_dir:
        mode = args.mode if args.mode else "custom"
        output_dir = Path(args.output_dir) / "tracks" / args.model_name / mode
        output_dir.mkdir(exist_ok=True, parents=True)
        check_output(
            generator,
            prompt[-1],
            args,
            output_dir,
            max_new_tokens=256,
            **generate_kwargs,
        )
