import io
import time
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import requests


# -----------------------------
# Image display helpers
# -----------------------------


def build_annotated_collage(
    images: Sequence[Image.Image],
    labels: Sequence[str],
    scores_by_metric: Dict[str, Sequence[Optional[float]]],
    cols: int = 4,
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """Draw a grid collage with per-image labels and metric annotations.

    - images: list of PIL images
    - labels: one label per image
    - scores_by_metric: mapping from metric name to list of per-image scores
    - cols: number of columns
    - figsize: matplotlib figure size
    """
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize[0], max(figsize[1], 3 * rows)), dpi=600
    )
    fig.patch.set_facecolor("black")

    axes = np.array(axes).reshape(rows, cols)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.imshow(img if img.mode != 'L' else img.convert('RGB'))
        ax.axis('off')
        text = f"{labels[idx].capitalize()}\n"
        for metric_name, values in scores_by_metric.items():
            val = values[idx] if idx < len(values) else None
            if val is None:
                continue
            text += f"{metric_name}: {val:.3f}\n"
        ax.set_title(text, fontsize=14, color='white')
    # Hide extras
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis('off')
    plt.tight_layout()
    plt.show()

def plot_image_grid(
    images: Sequence[Image.Image],
    titles: Optional[Sequence[str]] = None,
    rows: int = 1,
    cols: int = 1,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Plot images in a fixed rows x cols grid with optional titles."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=600)
    fig.patch.set_facecolor("black")

    axes_arr = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes_arr[r, c]
        if i < len(images):
            img = images[i]
            ax.imshow(img if img.mode != 'L' else img.convert('RGB'))
            if titles and i < len(titles):
                ax.set_title(titles[i], color='white')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def build_metric_demo_grid(
    images: List[Image.Image],
    labels: List[str],
    metric_name: str,
    metric_scores: List[float],
    cols: int = 4,
    figsize: Tuple[int, int] = (16, 8),
):
    """
    Helper to plot reference plus deformed images with labels and a single metric.

    Titles include the deformation name and the metric value rounded to 3 decimals.
    The first tile is always the reference image.
    """
    titles = []
    for label, score in zip(labels, metric_scores):
        titles.append(f"{label}\n{metric_name}: {score:.3f}")

    rows = int(np.ceil(len(images) / cols))
    plot_image_grid(images, titles=titles, rows=rows, cols=cols, figsize=figsize)

# -----------------------------
# Image deformations
# -----------------------------

def lower_resolution(img: Image.Image, factor: int = 16) -> Image.Image:
    size = img.size
    low_res_img = (
        img.resize(
            (max(1, size[0] // factor), max(1, size[1] // factor)),
            Image.Resampling.BILINEAR,
        ).resize(img.size, Image.Resampling.NEAREST)
    )
    return low_res_img


def add_noise(img: Image.Image, std: float = 120) -> Image.Image:
    img_arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, img_arr.shape)
    noisy_arr = np.clip(img_arr + noise, 0, 255)
    noisy_image = Image.fromarray(noisy_arr.astype(np.uint8))
    return noisy_image


def compress(img: Image.Image, quality: int = 1) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    compressed_img = Image.open(buffer).copy()
    buffer.close()
    return compressed_img


DEFORMATIONS = {
    'original': lambda img: img,
    'lower resolution': lambda img: lower_resolution(img, factor=8),
    'adding noise': lambda img: add_noise(img, std=30),
    'blurring': lambda img: img.filter(ImageFilter.GaussianBlur(radius=4)),
    'compression': lambda img: compress(img, quality=10),
    'black and white': lambda img: ImageOps.grayscale(img),
    'increasing saturation': lambda img: ImageEnhance.Color(img).enhance(factor=1.5),
    'reducing brightness': lambda img: ImageEnhance.Brightness(img).enhance(factor=0.5),
}

def apply_deformations(img: Image.Image, deformations: List[str]) -> List[Image.Image]:
    return [DEFORMATIONS[deformation](img) for deformation in deformations]

# -----------------------------
# Data loading helpers
# -----------------------------

def load_demo_image(
    run_internet: bool = True,
    fallback_size: Tuple[int, int] = (512, 512),
    fallback_color: Tuple[int, int, int] = (180, 160, 140),
) -> Image.Image:
    """Load a simple demo image.

    Attempts to fetch from picsum; on any failure or if ``run_internet`` is False,
    returns a solid RGB image as a safe fallback.
    """
    if run_internet:
        try:
            r = requests.get(
                'https://i.postimg.cc/9QS3ggW2/astronaut.png',
                timeout=15
            )
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", fallback_size, color=fallback_color)

def _get_url_image(
    url: str,
    timeout: float = 30.0,
    fallback_size: Tuple[int, int] = (512, 512),
    fallback_color: Tuple[int, int, int] = (127, 127, 127),
) -> Image.Image:
    """Fetch image by URL with fallback to solid RGB image."""
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Qlip.Algorithms-image-downloader"},
        )
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return Image.new("RGB", fallback_size, color=fallback_color)


# -----------------------------
# Dataset downloaders (Cats & Dogs)
# -----------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fetch_cat_image_urls(limit: int = 10) -> List[str]:
    response = requests.get(
        "https://api.thecatapi.com/v1/images/search",
        params={"limit": limit},
        timeout=20,
        headers={"User-Agent": "Qlip.Algorithms-image-downloader"},
    )
    response.raise_for_status()
    data = response.json()
    return [item.get("url") for item in data[:limit] if isinstance(item, dict) and item.get("url")]


def _fetch_dog_image_urls(limit: int = 10) -> List[str]:
    response = requests.get(
        f"https://dog.ceo/api/breeds/image/random/{limit}",
        timeout=20,
        headers={"User-Agent": "Qlip.Algorithms-image-downloader"},
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and isinstance(data.get("message"), list):
        return list(data["message"])  # type: ignore[index]
    return []


def _guess_extension_from_url(url: str) -> str:
    lower = url.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if ext in lower:
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def _download_image_to_path(url: str, output_path: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=40, headers={"User-Agent": "Qlip.Algorithms-image-downloader"}) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


def download_cats_and_dogs(
    repo_root: Optional[Union[str, Path]] = None,
    num_dogs: int = 11,
    num_cats: int = 10,
    sleep_sec: float = 0.2,
    target_size: Optional[Tuple[int, int]] = (256, 256),
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """Download small cats and dogs datasets to data/animals and return PIL images.

    - repo_root: project root directory (default: current working directory)
    - num_cats: number of cat images
    - num_dogs: number of dog images
    - sleep_sec: delay between downloads
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    cats_dir = root / "data" / "animals" / "cats"
    dogs_dir = root / "data" / "animals" / "dogs"
    _ensure_dir(cats_dir)
    _ensure_dir(dogs_dir)

    cat_urls = _fetch_cat_image_urls(limit=int(num_cats))
    dog_urls = _fetch_dog_image_urls(limit=int(num_dogs))

    for index, url in enumerate(cat_urls, start=1):
        ext = _guess_extension_from_url(url)
        out_path = cats_dir / f"cat_{index:02d}{ext}"
        _download_image_to_path(url, out_path)
        time.sleep(max(0.0, float(sleep_sec)))

    for index, url in enumerate(dog_urls, start=1):
        ext = _guess_extension_from_url(url)
        out_path = dogs_dir / f"dog_{index:02d}{ext}"
        _download_image_to_path(url, out_path)
        time.sleep(max(0.0, float(sleep_sec)))

    # Load back as PIL lists (and resize if needed)
    cat_imgs: List[Image.Image] = []
    for p in sorted(cats_dir.glob("cat_*")):
        try:
            img = Image.open(p).convert("RGB")
            if target_size is not None:
                img = img.resize(target_size, Image.Resampling.BILINEAR)
            cat_imgs.append(img)
        except Exception:
            continue

    dog_imgs: List[Image.Image] = []
    for p in sorted(dogs_dir.glob("dog_*")):
        try:
            img = Image.open(p).convert("RGB")
            if target_size is not None:
                img = img.resize(target_size, Image.Resampling.BILINEAR)
            dog_imgs.append(img)
        except Exception:
            continue

    return dog_imgs[:num_dogs], cat_imgs[:num_cats]
