import argparse
import json
import logging
import os
import timeit
from pathlib import Path
from typing import List, Optional

import torch

from elastic_models.diffusers import DiffusionPipeline
from diffusers import DiffusionPipeline as HFDiffusionPipeline
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

from gpu_monitor import GPUMemoryMonitor
from logger import _LOGGER_MAIN

logger = logging.getLogger(__name__)


# ------------------
# Load model
# ------------------
def get_pipeline(args):
    if args.mode == "original":
        pipe = HFDiffusionPipeline.from_pretrained(
            args.model_name,
            torch_dtype=args.dtype,
            cache_dir=args.cache_dir,
            token=args.hf_token,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            args.model_name,
            torch_dtype=args.dtype,
            cache_dir=args.cache_dir,
            token=args.hf_token,
            mode=args.mode,
            device_map=args.device
        )
    pipe.enable_vae_tiling()

    for model in pipe.components.values():
        if isinstance(model, torch.nn.Module):
            model.to(args.device)

    return pipe


def _run_mochi_pipeline(pipe, prompt, **kwargs):
    with torch.no_grad():
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(prompt=prompt, device=pipe.device)
        if prompt_attention_mask is not None and isinstance(
            prompt_attention_mask, torch.Tensor
        ):
            prompt_attention_mask = prompt_attention_mask.to(pipe.dtype)

        if negative_prompt_attention_mask is not None and isinstance(
            negative_prompt_attention_mask, torch.Tensor
        ):
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                pipe.dtype
            )

        prompt_embeds = prompt_embeds.to(pipe.dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(pipe.dtype)
        with torch.autocast("cuda", pipe.dtype, enabled=True):
            latents = pipe(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                output_type="latent",
                **kwargs,
            ).frames

        has_latents_mean = (
            hasattr(pipe.vae.config, "latents_mean")
            and pipe.vae.config.latents_mean is not None
        )
        has_latents_std = (
            hasattr(pipe.vae.config, "latents_std")
            and pipe.vae.config.latents_std is not None
        )

        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, 12, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(pipe.vae.config.latents_std)
                .view(1, 12, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = (
                latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
            )
        else:
            latents = latents / pipe.vae.config.scaling_factor

        with torch.autocast("cuda", pipe.dtype, enabled=False):
            video_frames = pipe.vae.decode(
                latents.to(pipe.vae.dtype), return_dict=False
            )[0]

        video_processor = VideoProcessor(
            vae_scale_factor=pipe.vae.config.scaling_factor
        )
        video_frames = video_processor.postprocess_video(video=video_frames)

    return video_frames


# ------------------
# Check output
# ------------------
def generate_content(model, prompts, args, folder, **kwargs):
    """Generate images or videos depending on the model type"""
    kwargs["generator"] = torch.Generator(device=args.device).manual_seed(args.seed)

    is_video_model = "mochi" in args.model_name.lower()

    if is_video_model:
        output_frames = _run_mochi_pipeline(model, prompts, **kwargs)
        for idx, (prompt, frames) in enumerate(zip(prompts, output_frames)):
            filename = f"{prompt.replace(' ', '_')}_{kwargs['width']}x{kwargs['height']}_{args.seed}_idx{idx}.mp4"
            export_to_video(frames, folder / filename, fps=30)
    else:
        output = model(prompt=prompts, **kwargs)
        # For image models
        for idx, (prompt, output_image) in enumerate(zip(prompts, output.images)):
            filename = f"{prompt.replace(' ', '_')}_{kwargs['width']}x{kwargs['height']}_{args.seed}_idx{idx}.png"
            output_image.save(folder / filename)


# ------------------
# Benchmarking
# ------------------
def benchmark_time(pipeline, prompt, number=3, repeat=3, **kwargs):
    is_video_model = "mochi" in pipeline.config._name_or_path.lower()

    if is_video_model:
        run_func = lambda: _run_mochi_pipeline(pipeline, prompt, **kwargs)
    else:
        run_func = lambda: pipeline(prompt=prompt, **kwargs)

    with torch.no_grad():
        runs = timeit.repeat(
            run_func,
            number=number,
            repeat=repeat,
            setup="import torch; torch.cuda.synchronize()",
        )
    return min(runs) / number


def benchmark(pipeline, prompt, number=1, repeat=5, include_memory=True, **kwargs):
    device = pipeline.device
    with GPUMemoryMonitor(device) as gpu_monitor:
        if not include_memory:
            gpu_monitor.stop()
        t = benchmark_time(pipeline, prompt, number=number, repeat=repeat, **kwargs)
    max_memory_usage = gpu_monitor.get_max_memory_usage()

    results = {
        "time": t,
        "batch_size": len(prompt),
        "width": kwargs["width"],
        "height": kwargs["height"],
    }

    if "num_frames" in kwargs:
        results["num_frames"] = kwargs["num_frames"]

    if include_memory:
        results["max_memory_usage"] = max_memory_usage

    return results


def benchmark_quality(pipeline, args, generate_kwargs):
    from qlip_algorithms.evaluation import (
        PartiPromptsEvaluator,
        DrawBenchEvaluator,
        CocoEvaluator,
    )

    results = {}

    for task_name in args.bench_tasks:
        if task_name == "partiprompts":
            evaluator = PartiPromptsEvaluator(
                pipeline=pipeline,
                cache_dir=args.cache_dir,
                group="challenge",
                num_samples=args.max_samples,
                generation_kwargs=generate_kwargs,
            )
            results[task_name] = evaluator.evaluate()
        elif task_name == "drawbench":
            evaluator = DrawBenchEvaluator(
                pipeline=pipeline,
                cache_dir=args.cache_dir,
                num_samples=args.max_samples,
                generation_kwargs=generate_kwargs,
            )
            results[task_name] = evaluator.evaluate()
        elif task_name == "coco":
            evaluator = CocoEvaluator(
                pipeline=pipeline,
                cache_dir=args.cache_dir,
                num_samples=args.max_samples,
                generation_kwargs=generate_kwargs,
            )
            results[task_name] = evaluator.evaluate()
        else:
            raise ValueError(f"Invalid benchmark task: {task_name}")

    return results


KWARGS_PER_MODEL = {
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "width": 1024,
        "height": 1024,
    },
    "black-forest-labs/FLUX.1-schnell": {
        "num_inference_steps": 4,
        "guidance_scale": 0,
        "width": 1024,
        "height": 1024,
    },
    "black-forest-labs/FLUX.1-dev": {
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "width": 1024,
        "height": 1024,
    },
    "genmo/mochi-1-preview": {
        "num_inference_steps": 28,
        "guidance_scale": 4.5,
        "num_frames": 19,
        "width": 848,
        "height": 480,
    },
}


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Image/Video Generation for Elastic Models"
    )
    parser.add_argument("--model_name", default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument(
        "--mode", default=None, choices=["original", "S", "M", "L", "XL", None]
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""), type=str)
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Directory to store cache files",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sizes", type=int, default=None, nargs=2)
    parser.add_argument(
        "--num_frames", type=int, default=None, help="Number of frames for video models"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="Big friendly dog")

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    parser.add_argument("--dtype", type=str, default=None, help="dtype")
    parser.add_argument(
        "--content_dir",
        type=str,
        default=None,
        help="Directory to save generated content",
    )
    parser.add_argument("--no_memory", action="store_true")
    parser.add_argument(
        "--bench_tasks",
        nargs="+",
        default=[],
        help="Tasks for quality benchmarks, e.g. `partiprompts`, `drawbench`, `coco`",
    )
    parser.add_argument(
        "--max_samples", type=int, default=1000, help="Maximum samples for evaluators"
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="File to save results"
    )
    args = parser.parse_args(args)

    args.device = torch.device(args.device)
    if args.sizes:
        args.sizes = tuple(args.sizes)  # Convert list to tuple

    if args.dtype is None:
        if args.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
            args.dtype = torch.float16
        else:
            args.dtype = torch.bfloat16
    else:
        args.dtype = getattr(torch, args.dtype)

    if not args.hf_token:
        logger.warning("HF_TOKEN environment variable is not set")

    return args


if __name__ == "__main__":
    args = parse_args()

    _LOGGER_MAIN.info(
        f"Loading model {args.model_name} in {args.mode} mode."
    )
    pipeline = get_pipeline(args)
    _LOGGER_MAIN.info(
        f"Model {args.model_name} in {args.mode} mode loaded successfully."
    )
    kwargs = KWARGS_PER_MODEL.get(args.model_name, {})

    if args.sizes:
        kwargs["width"], kwargs["height"] = args.sizes

    if args.num_frames:
        kwargs["num_frames"] = args.num_frames

    if "width" not in kwargs or "height" not in kwargs:
        raise ValueError(
            "Image sizes must be provided either through command line arguments or be present in KWARGS_PER_MODEL."
        )

    prompt = [args.prompt] * args.batch_size

    _LOGGER_MAIN.info(
        f"Starting latency benchmark for {args.model_name} in {args.mode} mode"
    )
    results = benchmark(
        pipeline,
        prompt,
        include_memory=not args.no_memory,
        **kwargs,
    )
    _LOGGER_MAIN.info(
        f"Latency benchmark for {args.model_name} in {args.mode} mode are ready:"
    )
    for key, value in results.items():
        _LOGGER_MAIN.info(f"{key}: {value}")
    _LOGGER_MAIN.info("Latency benchmarking completed.")

    quality_results = None
    if args.bench_tasks:
        _LOGGER_MAIN.info("Starting quality benchmarks: %s" % ", ".join(args.bench_tasks))
        quality_results = benchmark_quality(pipeline, args, kwargs)
        _LOGGER_MAIN.info("Quality benchmarks completed.")
        _LOGGER_MAIN.info("Results for quality benchmarks:")
        for task_name, task_res in quality_results.items():
            _LOGGER_MAIN.info(f"{task_name}: {task_res}")

    # print(f"Results for {args.mode} mode:")
    # print(results)

    if args.output_file:
        all_results = {
            "model_name": args.model_name,
            "mode": args.mode,
            "batch_size": args.batch_size,
            "inference_results": results,
        }

        if args.bench_tasks and quality_results is not None:
            all_results["quality_results"] = quality_results

        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        _LOGGER_MAIN.info(f"Results saved to {args.output_file}")

    if args.content_dir:
        mode = args.mode if args.mode else ""
        content_dir = Path(args.content_dir) / "content" / args.model_name / mode
        content_dir.mkdir(exist_ok=True, parents=True)
        generate_content(
            pipeline,
            prompt,
            args,
            content_dir,
            **kwargs,
        )
