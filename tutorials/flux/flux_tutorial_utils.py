"""
Utility functions for Flux ANNA analysis tutorial.
"""
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datasets import load_dataset


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset_prompts(dataset_name: str = "TempoFunk/webvid-10M",
                        max_samples: int = 1000, hf_token: str = None,
                        cache_dir: str = '/mount/huggingface_cache'):
    """Load prompts from dataset similar to run_vg.py."""
    try:
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(
            dataset_name,
            split='train',
            cache_dir=cache_dir,
            token=hf_token,
            trust_remote_code=True
        )

        # Extract prompts (adjust field name based on dataset)
        prompt_field = 'name' if 'name' in dataset.column_names else 'text'
        if prompt_field in dataset.column_names:
            prompts = [item[prompt_field] for item in dataset.select(range(min(max_samples, len(dataset))))]
            print(f"Loaded {len(prompts)} prompts from dataset")
            return prompts
        else:
            print(f"Warning: Could not find prompt field in dataset columns: {dataset.column_names}")
            return None
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None


def evaluate_configurations(analyser, results, output_dir):
    """Evaluate configurations by generating images."""
    print("Evaluating configurations with image generation...")

    # Evaluation prompts
    evaluation_prompts = [
        "A majestic lion standing on a rocky cliff at sunset",
        "A futuristic city skyline with flying cars and neon lights",
        "A beautiful garden with blooming flowers and butterflies",
        "An astronaut walking on the surface of an alien planet",
        "A vintage steam train crossing a stone bridge in the mountains",
        "A colorful hot air balloon floating over a green valley",
        "A magical forest with glowing mushrooms and fairy lights",
        "A cozy cabin in the woods with smoke coming from the chimney",
        "A bustling marketplace in an ancient city",
        "A serene lake reflecting snow-capped mountains at dawn"
    ]

    # Create eval_results directory
    eval_results_dir = os.path.join(output_dir, "eval_results")
    os.makedirs(eval_results_dir, exist_ok=True)

    for i, result in enumerate(results):
        constraint_value = result.constraint_value
        print(f"\nGenerating images for configuration {i+1} (constraint={constraint_value:.4f})...")

        # Apply configuration
        result.anna_config.apply(analyser.model)
        analyser.model.to(analyser.pipeline.device)

        # Create directories for this constraint
        constraint_dir = os.path.join(eval_results_dir, f"{constraint_value:.4f}")
        images_dir = os.path.join(constraint_dir, "evaluation_images")
        prompts_dir = os.path.join(constraint_dir, "evaluation_prompts")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(prompts_dir, exist_ok=True)

        # Save prompts for this constraint
        with open(os.path.join(prompts_dir, "prompts.txt"), 'w', encoding='utf-8') as f:
            for prompt in evaluation_prompts:
                f.write(f"{prompt}\n")

        # Generate images for each prompt
        generator = torch.Generator(device=analyser.pipeline.device).manual_seed(42)

        for j, prompt in enumerate(evaluation_prompts):
            try:
                with torch.no_grad():
                    result_image = analyser.pipeline(
                        prompt=prompt,
                        width=analyser.width,
                        height=analyser.height,
                        num_inference_steps=analyser.num_inference_steps,
                        guidance_scale=analyser.guidance_scale,
                        generator=generator,
                    ).images[0]

                # Save generated image
                image_path = os.path.join(images_dir, f'image_{j:03d}.png')
                result_image.save(image_path)

                print(f"Saved: {image_path}")

            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")
                continue

        # Remove configuration
        result.anna_config.remove()
        print(f"Generated images for configuration {constraint_value:.4f}")


def create_flux_model_configs():
    """Create configurations for different Flux models."""
    return {
        'flux-dev': {
            'model_name': 'black-forest-labs/FLUX.1-dev',
            'description': 'Flux Dev - High quality text-to-image generation',
            'recommended_steps': 28,
            'recommended_guidance': 3.5,
            'recommended_size': 1024
        },
        'flux-schnell': {
            'model_name': 'black-forest-labs/FLUX.1-schnell',
            'description': 'Flux Schnell - Fast text-to-image generation',
            'recommended_steps': 4,
            'recommended_guidance': 3.5,
            'recommended_size': 1024
        }
    }


def visualize_analysis_results(results):
    """
    Visualize the trade-off between model compression and quality degradation.

    Args:
        results: List of ANNA analysis results
    """
    # Extract constraint and objective values for visualization
    constraint_values = [result.constraint_value for result in results]
    real_loss_values = [result.real_loss_value for result in results]
    objective_loss_values = [result.objective_value.item() if torch.is_tensor(result.objective_value) else result.objective_value for result in results]

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Real Loss
    ax1.scatter(constraint_values, real_loss_values, c='blue', s=100, alpha=0.7, edgecolors='black', linewidth=1, label='Real Loss')
    ax1.plot(constraint_values, real_loss_values, 'b--', alpha=0.5)

    # Customize the first plot
    ax1.set_xlabel('Constraint Value (Model Size Ratio)', fontsize=12)
    ax1.set_ylabel('Real Loss Value (Quality Degradation)', fontsize=12)
    ax1.set_title('ANNA Analysis: Real Loss vs Model Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add annotations for key points on real loss plot
    min_real_idx = np.argmin(real_loss_values)
    min_constraint_idx = np.argmin(constraint_values)

    ax1.annotate(f'Best Quality\n({constraint_values[min_real_idx]:.3f}, {real_loss_values[min_real_idx]:.6f})',
                 xy=(constraint_values[min_real_idx], real_loss_values[min_real_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='green'))

    ax1.annotate(f'Highest Compression\n({constraint_values[min_constraint_idx]:.3f}, {real_loss_values[min_constraint_idx]:.6f})',
                 xy=(constraint_values[min_constraint_idx], real_loss_values[min_constraint_idx]),
                 xytext=(10, -20), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='red'))

    # Plot 2: Objective Loss
    ax2.scatter(constraint_values, objective_loss_values, c='red', s=100, alpha=0.7, edgecolors='black', linewidth=1, label='Objective Loss')
    ax2.plot(constraint_values, objective_loss_values, 'r--', alpha=0.5)

    # Customize the second plot
    ax2.set_xlabel('Constraint Value (Model Size Ratio)', fontsize=12)
    ax2.set_ylabel('Objective Loss Value', fontsize=12)
    ax2.set_title('ANNA Analysis: Objective Loss vs Model Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add annotations for key points on objective loss plot
    min_obj_idx = np.argmin(objective_loss_values)
    
    ax2.annotate(f'Best Objective\n({constraint_values[min_obj_idx]:.3f}, {objective_loss_values[min_obj_idx]:.6f})',
                 xy=(constraint_values[min_obj_idx], objective_loss_values[min_obj_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.show()

    print(f"Analysis shows {len(results)} configurations ranging from {min(constraint_values):.3f} to {max(constraint_values):.3f} model size ratio")
    print(f"Real Loss range: {min(real_loss_values):.6f} to {max(real_loss_values):.6f}")
    print(f"Objective Loss range: {min(objective_loss_values):.6f} to {max(objective_loss_values):.6f}")


def get_validation_prompts():
    """Get diverse validation prompts for comprehensive evaluation."""
    return [
        # Nature and landscapes
        "A majestic lion standing on a rocky cliff at sunset",
        "A serene lake reflecting snow-capped mountains at dawn",
        "A tropical beach with crystal clear water and palm trees",
        "A field of sunflowers stretching to the horizon",
        "A beautiful waterfall in a lush jungle",
        "A peaceful meadow with wildflowers and butterflies",
        
        # Urban and architecture
        "A futuristic city skyline with flying cars and neon lights",
        "A bustling marketplace in an ancient city",
        "A modern art gallery with abstract paintings on the walls",
        "A busy subway station during rush hour",
        "A medieval castle on top of a hill surrounded by fog",
        "A lighthouse standing on a rocky coast during a storm",
        
        # Fantasy and sci-fi
        "An astronaut walking on the surface of an alien planet",
        "A magical forest with glowing mushrooms and fairy lights",
        "A dragon flying over a fantasy landscape",
        "A space station orbiting Earth with astronauts working outside",
        "A magical portal opening in an enchanted forest",
        "A robot playing chess in a modern laboratory",
        
        # Transportation
        "A vintage steam train crossing a stone bridge in the mountains",
        "A colorful hot air balloon floating over a green valley",
        "A pirate ship sailing through a storm at sea",
        "A racing car speeding around a track",
        
        # Animals and nature
        "A group of penguins on an iceberg in Antarctica",
        "A beautiful garden with blooming flowers and butterflies",
        "A cozy cabin in the woods with smoke coming from the chimney",
        "A snow-covered village with warm lights in the windows",
        
        # Cultural and traditional
        "A traditional Japanese temple in cherry blossom season",
        "A cowboy riding a horse across the desert at sunset",
        "A scientist working in a high-tech laboratory",
        "A peaceful monastery in the mountains",
        
        # Portrait and people
        "A detailed portrait of a wise old wizard with a long beard",
        "A chef preparing an elaborate dish in a professional kitchen",
        "A ballet dancer leaping gracefully through the air",
        "An artist painting in a sunlit studio"
    ]


def visualize_quality_metrics(all_scores):
    """
    Visualize quality metrics (PSNR, SSIM, FID, CLIP) across different compression levels.
    
    Args:
        all_scores: Dictionary mapping constraint values to metric scores
    
    Returns:
        List of top configurations by combined quality score
    """
    import matplotlib.pyplot as plt
    
    # Extract data for plotting
    constraints = sorted(all_scores.keys())
    psnr_values = [all_scores[c].get('PSNR', 0) for c in constraints]
    ssim_values = [all_scores[c].get('SSIM', 0) for c in constraints]
    fid_values = [all_scores[c].get('FID', float('inf')) for c in constraints]
    clip_values = [all_scores[c].get('CLIP', 0) for c in constraints]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PSNR plot
    axes[0, 0].plot(constraints, psnr_values, 'b-o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Constraint (Model Size Ratio)')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].set_title('PSNR vs Model Compression', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
#     axes[0, 0].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Good quality threshold')
    axes[0, 0].legend()
    
    # SSIM plot
    axes[0, 1].plot(constraints, ssim_values, 'r-o', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Constraint (Model Size Ratio)')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].set_title('SSIM vs Model Compression', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
#     axes[0, 1].axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Good quality threshold')
    axes[0, 1].legend()
    
    # FID plot
    axes[1, 0].plot(constraints, fid_values, 'g-o', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Constraint (Model Size Ratio)')
    axes[1, 0].set_ylabel('FID')
    axes[1, 0].set_title('FID vs Model Compression', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
#     axes[1, 0].axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Good quality threshold')
    axes[1, 0].legend()
    axes[1, 0].invert_yaxis()  # Lower FID is better
    
    # CLIP Score plot
    axes[1, 1].plot(constraints, clip_values, 'm-o', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Constraint (Model Size Ratio)')
    axes[1, 1].set_ylabel('CLIP Score')
    axes[1, 1].set_title('CLIP Score vs Model Compression', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
#     axes[1, 1].axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Good alignment threshold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


def generate_example_results(analyser, results, pipeline, recommended_size, recommended_steps, recommended_guidance, show_examples=True):
    """
    Generate example images with different quantized configurations for comparison.

    Args:
        analyser: FluxAnalyser instance
        results: List of ANNA analysis results
        pipeline: Flux pipeline
        recommended_size: Image size for generation
        recommended_steps: Number of inference steps
        recommended_guidance: Guidance scale
        show_examples: Whether to show examples (default: True)
    """
    # Example prompts for testing
    example_prompts = [
        "A serene mountain landscape with a crystal clear lake reflecting the sky",
        "A futuristic city skyline at sunset with flying cars",
        "A detailed portrait of a wise old wizard with a long beard"
    ]

    if show_examples and len(results) >= 3:
        # Select configurations: highest compression, middle, and best quality
        sorted_results = sorted(results, key=lambda x: x.constraint_value)
        selected_configs = [
            (sorted_results[0], "Highest Compression"),
            (sorted_results[len(sorted_results)//2], "Balanced"),
            (sorted_results[-1], "Best Quality")
        ]

        print("Generating example images with different quantization levels...")
        print("This demonstrates how constraint values affect visual quality.")

        # Create a figure for comparison
        fig, axes = plt.subplots(len(selected_configs), len(example_prompts),
                                figsize=(15, 5*len(selected_configs)))
        if len(selected_configs) == 1:
            axes = [axes]

        for config_idx, (config, config_name) in enumerate(selected_configs):
            print(f"\n🔧 Testing {config_name} (constraint={config.constraint_value:.3f})...")

            # Apply configuration
            config.anna_config.apply(analyser.model)
            analyser.model.to(analyser.pipeline.device)

            for prompt_idx, prompt in enumerate(example_prompts):
                try:
                    print(f"  Prompt: \"{prompt}...\"")

                    # Generate image
                    with torch.no_grad():
                        result = pipeline(
                            prompt=prompt,
                            width=recommended_size,
                            height=recommended_size,
                            num_inference_steps=recommended_steps,
                            guidance_scale=recommended_guidance,
                            generator=torch.Generator(device=analyser.pipeline.device).manual_seed(42),
                        )

                    # Display image
                    ax = axes[config_idx][prompt_idx] if len(selected_configs) > 1 else axes[prompt_idx]
                    ax.imshow(result.images[0])
                    ax.set_title(f"{config_name}\n(constraint={config.constraint_value:.3f})\n{prompt[:30]}...",
                               fontsize=10, pad=10)
                    ax.axis('off')

                except Exception as e:
                    print(f"    Error: {e}")
                    ax = axes[config_idx][prompt_idx] if len(selected_configs) > 1 else axes[prompt_idx]
                    ax.text(0.5, 0.5, f'Error generating\nimage for:\n{config_name}',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')

            # Remove configuration
            config.anna_config.remove()

        plt.tight_layout()
        plt.show()

        print("\nComparison Summary:")
        for config, config_name in selected_configs:
            print(f"  • {config_name}: constraint={config.constraint_value:.3f}, objective={config.objective_value:.6f}")

        print("\nKey Observations:")
        print("  • Higher compression (lower constraint) = smaller model size but potentially lower quality")
        print("  • Lower compression (higher constraint) = larger model size but better quality preservation")
        print("  • The optimal balance depends on your specific use case and quality requirements")

    else:
        print("Skipping example visualization. Set show_examples=True and ensure you have sufficient configurations.")

                
def show_images(anna_image, full_quant_image):
    import matplotlib.pyplot as plt

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display optimized quantized result
    ax1.imshow(anna_image.images[0])
    ax1.set_title('Anna Config (INT8, min constraint)', fontsize=10)
    ax1.axis('off')

    # Display full quantization result
    ax2.imshow(full_quant_image.images[0])
    ax2.set_title('Full Model Quantization (INT8)', fontsize=10)
    ax2.axis('off')

    # Add overall title
    plt.suptitle('Comparison: Anna vs Full Quantization', fontsize=12, y=1.05)
    plt.tight_layout()
    plt.show()
