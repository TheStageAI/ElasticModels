import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from typing import Dict, Any, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import DataLoader, Dataset

from lm_eval import tasks, evaluator
from lm_eval.models.huggingface import HFLM

import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CalibrationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_benchmarks(
    model, tokenizer, task_names, 
    device, dtype=torch.float32, batch_size=1
):
    """Evaluate the model on specified benchmark tasks."""
    with torch.nn.utils.parametrize.cached():
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        out = {}
        for task_name in task_names:
            task_dict = tasks.get_task_dict([task_name])
            metrics = evaluator.evaluate(lm=lm, task_dict=task_dict)['results']
        
            accuracy = 0
            counter = 1
            for key, val in metrics.items():
                accuracy += val['acc,none']
                counter += 1
            out[task_name] = accuracy / counter
    
    return out


class ActivationStatsCollector:
    """
    Collects per-channel activation statistics for module *inputs* (by default),
    across multiple forward passes. Works for Linear (2D inputs) and Conv2d-like
    modules (4D inputs). Can be extended easily for other modules.

    Stats collected per channel: count, mean, M2 (for variance), min, max.
    Uses a stable online merge formula so you can accumulate across batches.
    """

    def __init__(self, modules_to_track=(nn.Linear,), collect_outputs=False):
        """
        modules_to_track: tuple of module classes to attach hooks to.
        collect_outputs: if True, collect statistics for module outputs instead of inputs.
        """
        self.modules_to_track = modules_to_track
        self.collect_outputs = collect_outputs
        self._handles = []
        self.stats_by_name: Dict[str, Dict[str, torch.Tensor]] = {}
        self.module_names: Dict[nn.Module, str] = {}
        self._registered = False

    def register_hooks(self, model: nn.Module):
        if self._registered:
            return
        for name, module in model.named_modules():
            if isinstance(module, self.modules_to_track):
                self.module_names[module] = name
                handle = module.register_forward_hook(self._make_hook(name, module))
                self._handles.append(handle)
        self._registered = True

    def _make_hook(self, name: str, module: nn.Module):
        def hook(mod, inp: Tuple[Any], out):
            tensor = out if self.collect_outputs else inp[0]
            tensor = tensor.abs()
            t = tensor.detach().cpu()
            if t.dim() == 3:
                batch, seq, features = t.shape
                samples = t.reshape(-1, features)  # (N_samples, C)
            else:
                if t.shape[-1] == 1:
                    return
                samples = t.reshape(-1, t.shape[-1])

            if samples.numel() == 0:
                return

            if name not in self.stats_by_name:
                C = samples.shape[1]
                self.stats_by_name[name] = {
                    "count": torch.tensor(0.0),
                    "abs_max": torch.full((C,), float("-inf")),
                }

            entry = self.stats_by_name[name]

            batch_count = float(samples.shape[0])
            batch_mean = samples.mean(dim=0)  # shape (C,)
            batch_var = samples.var(dim=0, unbiased=False)  # population variance
            batch_M2 = batch_var * batch_count  # M2 = sum((x - mean)^2)
            batch_min = samples.min(dim=0).values
            batch_max = samples.max(dim=0).values

            n = float(entry["count"].item())
            if n == 0.0:
                entry["abs_max"] = torch.maximum(entry["abs_max"], batch_max)
            else:
                m = batch_count
                total = n + m
                entry["count"] = torch.tensor(total)
                entry["abs_max"] = torch.maximum(entry["abs_max"], batch_max)

        return hook

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._registered = False

    def reset(self):
        self.stats_by_name = {}
        self.module_names = {}
        self.remove_hooks()

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        """
        out = {}
        for name, st in self.stats_by_name.items():
            count = float(st["count"].item())
            out[name] = st["abs_max"].clone().cpu()
        return out

            
def get_calibration_data(tokenizer):
    # Load WikiText dataset for calibration
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="train", 
        cache_dir="/mount/huggingface_cache"
    )

    # Prepare calibration data
    def prepare_calibration_data(dataset, tokenizer, max_length=512, num_samples=1000):
        """Prepare calibration dataset for quantization."""
        # Filter out empty texts
        texts = [text for text in dataset['text'] if len(text.strip()) > 50]
        # Take subset for calibration
        if len(texts) > num_samples:
            texts = texts[:num_samples]
        # Tokenize texts
        calibration_data = []
        for text in tqdm(texts[:100], desc="Tokenizing calibration data"):  # Use smaller subset for demo
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )
            calibration_data.append({
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0)
            })

        return calibration_data

    calibration_data = prepare_calibration_data(dataset, tokenizer)
    print(f"Prepared {len(calibration_data)} calibration samples")
    
    # Create DataLoader for calibration
    calib_dataset = CalibrationDataset(calibration_data)
    calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=False)
    
    return calib_loader
