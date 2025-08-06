import argparse
import logging
import os
import random
import timeit
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor
from transformers.pipelines import pipeline
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration as HFWhisperForConditionalGeneration,
)
from transformers.utils import logging as hf_logging

from gpu_monitor import GPUMemoryMonitor
from elastic_models.transformers import WhisperForConditionalGeneration

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
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model_name,
    #     cache_dir=args.cache_dir,
    #     padding_side='left',
    # )

    if args.mode == "original":
        model = HFWhisperForConditionalGeneration.from_pretrained(
            args.model_name,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            torch_dtype=args.dtype,
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_name,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            torch_dtype=args.dtype,
            mode=args.mode,
            device_map=args.device
        )

    model.eval()

    generator = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=args.device,
    )

    return generator


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
    batch_size = generate_kwargs.pop("batch_size", 1)
    runs = timeit.repeat(
        run_cuda_synchronize(
            lambda: generator(
                prompt, batch_size=batch_size, generate_kwargs=generate_kwargs
            )
        ),
        number=number,
        repeat=repeat,
        setup="import torch; torch.cuda.synchronize()",
    )
    return min(runs) / number


def benchmark_time(
    generator, samples, number=1, repeat=5, new_tokens=10, **generate_kwargs
):
    first = time_generation(
        generator,
        samples,
        number=number,
        repeat=repeat,
        max_new_tokens=1,
        **generate_kwargs,
    )

    fastest = time_generation(
        generator,
        samples,
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
    samples,
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
            samples,
            number=number,
            repeat=repeat,
            new_tokens=new_tokens,
            **generate_kwargs,
        )
    max_memory_usage = gpu_monitor.get_max_memory_usage()

    results = {
        "tps": t,
        "ttft": f,
        "batch_size": len(samples),
    }
    if include_memory:
        results["max_memory_usage"] = max_memory_usage

    return results


# -----------------
# Check output
# -----------------


def check_output(generator, samples, args, **generate_kwargs):
    if not isinstance(samples, list):
        samples = [samples]

    batch_size = generate_kwargs.pop("batch_size", 1)

    set_seed(args.seed)
    outputs = generator(samples, batch_size=batch_size, generate_kwargs=generate_kwargs)

    for idx, output in enumerate(outputs):
        output = output["text"].strip()
        print(f"\n# Output {idx}:\n{output}")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark text generation for Elastic Models"
    )
    parser.add_argument("--model_name", default="openai/whisper-large-v3")
    parser.add_argument(
        "--mode", default="original", choices=["original", "S", "M", "L", "XL", "XXL"]
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""), type=str)
    parser.add_argument(
        "--cache_dir",
        default="/mnt/rnd/huggingface_cache",
        type=str,
        help="Directory to store cache files",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check_output", action="store_true")

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default=torch.float16, help="dtype")
    parser.add_argument("--no_memory", action="store_true")
    args = parser.parse_args(args)
    args.device = torch.device(args.device)

    if not args.hf_token:
        logger.warning("HF_TOKEN environment variable is not set")

    return args


if __name__ == "__main__":
    args = parse_args()

    generator = get_generator(args)
    generate_kwargs = {"num_beams": 1, "disable_compile": True}
    if args.mode != "original":
        generate_kwargs["cache_implementation"] = "flexi-static"

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    samples = [s["audio"]["array"] for s in ds.select(range(args.batch_size))]
    # for sample in samples:
    #     sample['raw'] = sample['array']

    results = benchmark(
        generator,
        samples,
        batch_size=args.batch_size,
        **generate_kwargs,
        include_memory=not args.no_memory,
    )
    print(f"Results for {args.mode} mode:")
    print(results)

    if args.check_output:
        check_output(
            generator,
            samples,
            args,
            batch_size=args.batch_size,
            **generate_kwargs,
        )
