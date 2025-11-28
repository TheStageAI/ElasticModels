"""
Flux Caching Tutorial Utilities

This module contains helper functions for the Flux caching tutorial,
including compatibility patches for older diffusers versions.
"""

import sys
import time
import types
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Any


# ============================================================================
# Diffusers Compatibility Patches
# ============================================================================

def setup_diffusers_compatibility():
    """
    Setup compatibility patches for cache-dit with older diffusers versions.

    cache-dit tries to import several model classes that don't exist in older diffusers.
    We create dummy classes to prevent import errors (they won't be used for Flux).
    """
    print("Checking diffusers compatibility...")

    import diffusers

    # List of all classes that cache-dit might try to import
    FAKE_CLASSES = [
        ('ChromaTransformer2DModel', 'diffusers'),
        ('HiDreamImageTransformer2DModel', 'diffusers'),
        ('QwenImageTransformer2DModel', 'diffusers'),
    ]

    FAKE_SUBMODULES = {
        'diffusers.models.transformers.transformer_chroma': [
            'ChromaTransformerBlock',
            'ChromaSingleTransformerBlock',
            'Transformer2DModelOutput',
        ],
        'diffusers.models.transformers.transformer_hidream': [
            'HiDreamTransformerBlock',
            'HiDreamSingleTransformerBlock',
            'Transformer2DModelOutput',
        ],
        'diffusers.models.transformers.transformer_hidream_image': [
            'HiDreamBlock',
            'HiDreamImageTransformerBlock',
            'HiDreamImageSingleTransformerBlock',
            'Transformer2DModelOutput',
        ],
        'diffusers.models.transformers.transformer_qwenimage': [
            'QwenImageTransformerBlock',
            'Transformer2DModelOutput',
        ],
    }

    # Create base dummy class
    class FakeDiffusersClass:
        """Dummy class for compatibility with older diffusers versions."""
        pass

    # Patch missing classes in main diffusers module
    patched_classes = []
    for class_name, module_name in FAKE_CLASSES:
        try:
            # Try to import - if it exists, no need to patch
            exec(f"from {module_name} import {class_name}")
        except ImportError:
            # Create and inject fake class
            fake_class = type(class_name, (FakeDiffusersClass,), {
                '__doc__': f'Fake {class_name} for compatibility'
            })
            setattr(diffusers, class_name, fake_class)
            patched_classes.append(class_name)

    if patched_classes:
        print(f"⚠ Patched missing classes: {', '.join(patched_classes)}")

    # Patch missing submodules
    patched_modules = []
    for module_path, class_names in FAKE_SUBMODULES.items():
        if module_path not in sys.modules:
            # Create fake submodule with full path name
            fake_module = types.ModuleType(module_path)

            # Add fake classes to submodule
            for class_name in class_names:
                fake_class = type(class_name, (FakeDiffusersClass,), {
                    '__doc__': f'Fake {class_name} for compatibility'
                })
                setattr(fake_module, class_name, fake_class)

            # Register in sys.modules with full path
            sys.modules[module_path] = fake_module
            patched_modules.append(module_path.split('.')[-1])

    if patched_modules:
        print(f"⚠ Patched missing modules: {', '.join(patched_modules)}")

    if not patched_classes and not patched_modules:
        print("✓ All required classes found (no patching needed)")
    else:
        print("✓ Compatibility patches applied successfully")


# ============================================================================
# Core Helper Functions
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_elastic_pipeline(
    model_name: str,
    mode: str = 'original',
    model_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    hf_cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    device: str = 'cuda'
):
    """
    Load Elastic Flux pipeline in specified mode.

    Args:
        model_name: HuggingFace model name
        mode: Elastic mode ('original', 'XL', 'S')
        model_path: Path to custom elastic model weights
        dtype: Model precision
        hf_cache_dir: HuggingFace cache directory
        hf_token: HuggingFace authentication token
        device: Device to load model on

    Returns:
        Loaded pipeline
    """
    from elastic_models.diffusers import DiffusionPipeline as ElasticDiffusionPipeline

    print(f"\n{'='*60}")
    print(f"Loading Elastic Flux pipeline - Mode: {mode}")
    print(f"{'='*60}")

    if mode == 'original':
        pipeline = ElasticDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            cache_dir=hf_cache_dir,
            token=hf_token,
            device_map=device,
        )
    else:
        pipeline = ElasticDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            cache_dir=hf_cache_dir,
            token=hf_token,
            mode=mode,
            device_map=device,
            __model_path=model_path,
        )

    pipeline = pipeline.to(device)
    print(f"✓ Pipeline loaded successfully")
    return pipeline


def enable_dualcache(pipeline, mode: str = 'conservative'):
    """
    Enable DualCache on pipeline.

    Args:
        pipeline: Diffusion pipeline
        mode: Caching mode ('aggressive' or 'conservative')
    """
    import cache_dit
    from cache_dit import BasicCacheConfig, BlockAdapter, ForwardPattern

    print(f"\nEnabling DualCache ({mode} mode)...")

    if mode == 'aggressive':
        Fn = 1
        Bn = 0
        max_warmup_steps = 8
        max_continuous_cached_steps = 10
        residual_diff_threshold = 0.2
    else:  # conservative
        Fn = 4
        Bn = 0
        max_warmup_steps = 8
        max_continuous_cached_steps = 3
        residual_diff_threshold = 0.05

    cache_config = BasicCacheConfig(
        Fn_compute_blocks=Fn,
        Bn_compute_blocks=Bn,
        max_warmup_steps=max_warmup_steps,
        max_cached_steps=-1,
        max_continuous_cached_steps=max_continuous_cached_steps,
        residual_diff_threshold=residual_diff_threshold,
    )

    # Check if blocks are compiled
    first_block = pipeline.transformer.transformer_blocks[0]
    is_compiled = hasattr(first_block, '__wrapped__') or 'Compiled' in type(first_block).__name__

    cache_dit.enable_cache(
        BlockAdapter(
            pipe=pipeline,
            transformer=pipeline.transformer,
            blocks=[
                pipeline.transformer.transformer_blocks,
                pipeline.transformer.single_transformer_blocks,
            ],
            forward_pattern=[
                ForwardPattern.Pattern_1,
                ForwardPattern.Pattern_3,
            ],
            check_forward_pattern=not is_compiled,
        ),
        cache_config=cache_config
    )

    print(f"✓ DualCache enabled: Fn={Fn}, rdt={residual_diff_threshold}, max_continuous={max_continuous_cached_steps}")


def disable_cache(pipeline):
    """Disable cache on pipeline."""
    import cache_dit
    cache_dit.disable_cache(pipeline)
    print("✓ Cache disabled")


def generate_and_time(
    pipeline,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    Generate image and measure time.

    Args:
        pipeline: Diffusion pipeline
        prompt: Text prompt
        width: Image width
        height: Image height
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed
        device: Device

    Returns:
        Tuple of (image, elapsed_time)
    """
    set_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    torch.cuda.synchronize()
    start_time = time.time()

    result = pipeline(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    return result.images[0], elapsed


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_comparison(
    results_dict: Dict[str, Any],
    mode_name: str,
    test_prompts: List[str],
    output_dir: Path,
    prompt_idx: int = 0
):
    """
    Visualize images from different caching strategies.

    Args:
        results_dict: Dictionary with results for each config
        mode_name: Mode name (e.g., 'XL', 'S')
        test_prompts: List of test prompts
        output_dir: Output directory for saving plots
        prompt_idx: Index of prompt to visualize
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    configs = ['no_cache', 'aggressive', 'conservative']
    titles = ['No Cache', 'Aggressive Cache', 'Conservative Cache']

    for i, (config, title) in enumerate(zip(configs, titles)):
        image = results_dict[config]['images'][prompt_idx]
        avg_time = results_dict[config]['avg_time']

        axes[i].imshow(image)
        axes[i].axis('off')

        if 'speedup' in results_dict[config]:
            speedup = results_dict[config]['speedup']
            axes[i].set_title(f"{title}\n{avg_time:.2f}s ({speedup:.2f}x speedup)", fontsize=12)
        else:
            axes[i].set_title(f"{title}\n{avg_time:.2f}s (baseline)", fontsize=12)

    plt.suptitle(f"Mode {mode_name}: {test_prompts[prompt_idx]}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"comparison_{mode_name.lower()}_prompt{prompt_idx}.png", dpi=150, bbox_inches='tight')
    plt.show()


def create_performance_charts(
    results_xl: Dict[str, Any],
    results_s: Dict[str, Any],
    output_dir: Path
):
    """
    Create performance comparison bar charts.

    Args:
        results_xl: Results dictionary for XL mode
        results_s: Results dictionary for S mode
        output_dir: Output directory for saving plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # XL performance
    xl_times = [
        results_xl['no_cache']['avg_time'],
        results_xl['aggressive']['avg_time'],
        results_xl['conservative']['avg_time']
    ]
    xl_labels = ['No Cache', 'Aggressive', 'Conservative']
    xl_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    bars1 = ax1.bar(xl_labels, xl_times, color=xl_colors, alpha=0.8)
    ax1.set_ylabel('Average Time (seconds)', fontsize=12)
    ax1.set_title('Mode XL: Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # S performance
    s_times = [
        results_s['no_cache']['avg_time'],
        results_s['aggressive']['avg_time'],
        results_s['conservative']['avg_time']
    ]
    s_labels = ['No Cache', 'Aggressive', 'Conservative']
    s_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    bars2 = ax2.bar(s_labels, s_times, color=s_colors, alpha=0.8)
    ax2.set_ylabel('Average Time (seconds)', fontsize=12)
    ax2.set_title('Mode S: Performance Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_performance_summary(
    results_xl: Dict[str, Any],
    results_s: Dict[str, Any]
):
    """
    Print performance summary table for all experiments.

    Args:
        results_xl: Results dictionary for XL mode
        results_s: Results dictionary for S mode
    """
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    # XL Mode summary
    print("\n" + "-"*80)
    print("MODE XL:")
    print("-"*80)
    print(f"{'Config':<20} {'Avg Time (s)':<15} {'Speedup':<10} {'Cache Hit Rate':<15}")
    print("-"*80)

    xl_nocache_time = results_xl['no_cache']['avg_time']
    print(f"{'No Cache':<20} {xl_nocache_time:<15.2f} {'1.00x':<10} {'N/A':<15}")

    for config_name in ['aggressive', 'conservative']:
        config = results_xl[config_name]
        avg_time = config['avg_time']
        speedup = config['speedup']

        # Extract cache stats if available
        if 'cache_stats' in config and config['cache_stats']:
            cache_stats = config['cache_stats']
            # Calculate average hit rate across all blocks
            hit_rates = []

            # Handle different cache_stats formats
            if isinstance(cache_stats, list):
                # New format: list of CacheStats objects
                for stats_obj in cache_stats:
                    if hasattr(stats_obj, 'cached_steps') and hasattr(stats_obj, 'residual_diffs'):
                        total_steps = len(stats_obj.residual_diffs)
                        cached_count = len(stats_obj.cached_steps)
                        if total_steps > 0:
                            hit_rate = cached_count / total_steps
                            hit_rates.append(hit_rate)
            elif isinstance(cache_stats, dict):
                # Old format: dict with hit_rate
                for block_stats in cache_stats.values():
                    if isinstance(block_stats, dict) and 'hit_rate' in block_stats:
                        hit_rates.append(block_stats['hit_rate'])

            avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0
            hit_rate_str = f"{avg_hit_rate:.1%}"
        else:
            hit_rate_str = "N/A"

        config_label = config_name.capitalize()
        print(f"{config_label:<20} {avg_time:<15.2f} {speedup:<10.2f}x {hit_rate_str:<15}")

    # S Mode summary
    print("\n" + "-"*80)
    print("MODE S:")
    print("-"*80)
    print(f"{'Config':<20} {'Avg Time (s)':<15} {'Speedup':<10} {'Cache Hit Rate':<15}")
    print("-"*80)

    s_nocache_time = results_s['no_cache']['avg_time']
    print(f"{'No Cache':<20} {s_nocache_time:<15.2f} {'1.00x':<10} {'N/A':<15}")

    for config_name in ['aggressive', 'conservative']:
        config = results_s[config_name]
        avg_time = config['avg_time']
        speedup = config['speedup']

        # Extract cache stats if available
        if 'cache_stats' in config and config['cache_stats']:
            cache_stats = config['cache_stats']
            # Calculate average hit rate across all blocks
            hit_rates = []

            # Handle different cache_stats formats
            if isinstance(cache_stats, list):
                # New format: list of CacheStats objects
                for stats_obj in cache_stats:
                    if hasattr(stats_obj, 'cached_steps') and hasattr(stats_obj, 'residual_diffs'):
                        total_steps = len(stats_obj.residual_diffs)
                        cached_count = len(stats_obj.cached_steps)
                        if total_steps > 0:
                            hit_rate = cached_count / total_steps
                            hit_rates.append(hit_rate)
            elif isinstance(cache_stats, dict):
                # Old format: dict with hit_rate
                for block_stats in cache_stats.values():
                    if isinstance(block_stats, dict) and 'hit_rate' in block_stats:
                        hit_rates.append(block_stats['hit_rate'])

            avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0
            hit_rate_str = f"{avg_hit_rate:.1%}"
        else:
            hit_rate_str = "N/A"

        config_label = config_name.capitalize()
        print(f"{config_label:<20} {avg_time:<15.2f} {speedup:<10.2f}x {hit_rate_str:<15}")

    # Overall best configurations
    print("\n" + "-"*80)
    print("OVERALL ANALYSIS:")
    print("-"*80)

    # Find fastest configuration
    all_configs = [
        ("XL No Cache", xl_nocache_time, 1.0),
        ("XL Aggressive", results_xl['aggressive']['avg_time'], results_xl['aggressive']['speedup']),
        ("XL Conservative", results_xl['conservative']['avg_time'], results_xl['conservative']['speedup']),
        ("S No Cache", s_nocache_time, 1.0),
        ("S Aggressive", results_s['aggressive']['avg_time'], results_s['aggressive']['speedup']),
        ("S Conservative", results_s['conservative']['avg_time'], results_s['conservative']['speedup']),
    ]

    fastest = min(all_configs, key=lambda x: x[1])
    print(f"Fastest configuration: {fastest[0]} - {fastest[1]:.2f}s")

    # Best speedup from caching
    xl_best_speedup = max(results_xl['aggressive']['speedup'], results_xl['conservative']['speedup'])
    s_best_speedup = max(results_s['aggressive']['speedup'], results_s['conservative']['speedup'])

    print(f"\nBest XL speedup from caching: {xl_best_speedup:.2f}x")
    print(f"Best S speedup from caching: {s_best_speedup:.2f}x")

    # Recommendations
    print("\nRecommendations:")
    print("  • For maximum speed: S + Aggressive Cache")
    print("  • For quality-critical: XL + Conservative Cache")
    print("  • For balanced: XL + Aggressive Cache or S + Conservative Cache")
    print("="*80 + "\n")


def save_results(
    results_xl: Dict[str, Any],
    results_s: Dict[str, Any],
    output_dir: Path,
    model_name: str,
    width: int,
    height: int,
    num_inference_steps: int,
    test_prompts: List[str]
):
    """
    Save all results to disk including images, stats, and config.

    Args:
        results_xl: Results dictionary for XL mode
        results_s: Results dictionary for S mode
        output_dir: Output directory
        model_name: Model name used
        width: Image width
        height: Image height
        num_inference_steps: Number of inference steps
        test_prompts: List of test prompts
    """
    import json

    print(f"\nSaving results to {output_dir}...")

    # Create subdirectories
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)

    # Save images
    configs = ['no_cache', 'aggressive', 'conservative']

    for mode_name, results in [('xl', results_xl), ('s', results_s)]:
        for config in configs:
            for i, image in enumerate(results[config]['images']):
                filename = f"{mode_name}_{config}_prompt{i}.png"
                image.save(images_dir / filename)

    print(f"✓ Saved {len(test_prompts) * 2 * 3} images to {images_dir}")

    # Save performance statistics as JSON
    stats = {
        'model_name': model_name,
        'config': {
            'width': width,
            'height': height,
            'num_inference_steps': num_inference_steps,
            'test_prompts': test_prompts,
        },
        'results': {
            'xl': {
                'no_cache': {
                    'avg_time': results_xl['no_cache']['avg_time'],
                    'times': results_xl['no_cache']['times'],
                },
                'aggressive': {
                    'avg_time': results_xl['aggressive']['avg_time'],
                    'times': results_xl['aggressive']['times'],
                    'speedup': results_xl['aggressive']['speedup'],
                },
                'conservative': {
                    'avg_time': results_xl['conservative']['avg_time'],
                    'times': results_xl['conservative']['times'],
                    'speedup': results_xl['conservative']['speedup'],
                },
            },
            's': {
                'no_cache': {
                    'avg_time': results_s['no_cache']['avg_time'],
                    'times': results_s['no_cache']['times'],
                },
                'aggressive': {
                    'avg_time': results_s['aggressive']['avg_time'],
                    'times': results_s['aggressive']['times'],
                    'speedup': results_s['aggressive']['speedup'],
                },
                'conservative': {
                    'avg_time': results_s['conservative']['avg_time'],
                    'times': results_s['conservative']['times'],
                    'speedup': results_s['conservative']['speedup'],
                },
            },
        }
    }

    stats_file = output_dir / 'performance_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Saved performance statistics to {stats_file}")

    # Save summary report as text
    report_file = output_dir / 'summary_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FLUX CACHING TUTORIAL - RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: {model_name}\n")
        f.write(f"Image size: {width}x{height}\n")
        f.write(f"Inference steps: {num_inference_steps}\n")
        f.write(f"Number of test prompts: {len(test_prompts)}\n\n")

        f.write("Test Prompts:\n")
        for i, prompt in enumerate(test_prompts):
            f.write(f"  {i+1}. {prompt}\n")
        f.write("\n")

        f.write("-"*80 + "\n")
        f.write("MODE XL RESULTS:\n")
        f.write("-"*80 + "\n")
        f.write(f"No Cache:      {results_xl['no_cache']['avg_time']:.2f}s (baseline)\n")
        f.write(f"Aggressive:    {results_xl['aggressive']['avg_time']:.2f}s ({results_xl['aggressive']['speedup']:.2f}x speedup)\n")
        f.write(f"Conservative:  {results_xl['conservative']['avg_time']:.2f}s ({results_xl['conservative']['speedup']:.2f}x speedup)\n\n")

        f.write("-"*80 + "\n")
        f.write("MODE S RESULTS:\n")
        f.write("-"*80 + "\n")
        f.write(f"No Cache:      {results_s['no_cache']['avg_time']:.2f}s (baseline)\n")
        f.write(f"Aggressive:    {results_s['aggressive']['avg_time']:.2f}s ({results_s['aggressive']['speedup']:.2f}x speedup)\n")
        f.write(f"Conservative:  {results_s['conservative']['avg_time']:.2f}s ({results_s['conservative']['speedup']:.2f}x speedup)\n\n")

        f.write("-"*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("-"*80 + "\n")
        f.write("1. Caching provides significant speedups for both XL and S modes\n")
        f.write("2. Aggressive caching offers higher speedups but may trade some quality\n")
        f.write("3. Conservative caching provides balanced performance with better quality preservation\n")
        f.write("4. S mode + Aggressive cache gives best overall performance\n\n")

        f.write("="*80 + "\n")

    print(f"✓ Saved summary report to {report_file}")
    print(f"\n✓ All results saved successfully!")
