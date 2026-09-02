#!/usr/bin/env python3
"""SatQuery AI — LoRA Fine-Tuning on BigEarthNet-MM.

Domain-adapts a Vision-Language Model to remote-sensing imagery using
LoRA (Low-Rank Adaptation) on the BigEarthNet multi-modal dataset.

Usage:
    python training/train_bigearthnet_lora.py \
        --model_name Salesforce/blip2-opt-2.7b \
        --data_dir data/datasets/bigearthnet \
        --output_dir checkpoints/lora_bigearthnet \
        --epochs 5 \
        --lora_rank 16 \
        --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── BigEarthNet Dataset ──────────────────────────────────────────────────────

# BigEarthNet-19 class labels (simplified)
BEN_LABELS = [
    "Urban fabric", "Industrial/commercial", "Arable land",
    "Permanent crops", "Pastures", "Complex cultivation",
    "Agriculture + vegetation", "Broad-leaved forest",
    "Coniferous forest", "Mixed forest", "Natural grassland",
    "Moors/heathland", "Sclerophyllous vegetation",
    "Transitional woodland", "Beaches/dunes/sands",
    "Inland wetlands", "Coastal wetlands", "Inland waters",
    "Marine waters",
]


class BigEarthNetDataset(Dataset):
    """PyTorch dataset for BigEarthNet-MM.

    Loads Sentinel-2 patches and their multi-label annotations.
    Converts images to RGB for VLM input and formats labels as
    text descriptions for caption-style training.

    Args:
        data_dir: Path to BigEarthNet dataset directory.
        split: Train/val/test split.
        max_samples: Maximum number of samples to load (for debugging).
        image_size: Resize images to this size.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 224,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.samples = self._load_samples(max_samples)
        logger.info(
            f"BigEarthNet dataset loaded: {len(self.samples)} samples ({split})"
        )

    def _load_samples(self, max_samples: Optional[int]) -> List[Dict]:
        """Discover and load sample metadata."""
        samples = []

        # BigEarthNet structure: each patch is a directory with band TIFs + labels.json
        if not self.data_dir.exists():
            logger.warning(
                f"Data directory not found: {self.data_dir}. "
                f"Using synthetic data for demonstration."
            )
            return self._generate_synthetic_samples(max_samples or 100)

        patch_dirs = sorted(self.data_dir.iterdir())
        if max_samples:
            patch_dirs = patch_dirs[:max_samples]

        for patch_dir in patch_dirs:
            if not patch_dir.is_dir():
                continue

            label_file = patch_dir / "labels_metadata.json"
            if not label_file.exists():
                continue

            try:
                with open(label_file) as f:
                    metadata = json.load(f)
                labels = metadata.get("labels", [])
            except Exception:
                labels = []

            samples.append({
                "patch_dir": str(patch_dir),
                "labels": labels,
                "text": self._labels_to_text(labels),
            })

        return samples

    def _generate_synthetic_samples(self, n: int) -> List[Dict]:
        """Generate synthetic training samples when real data isn't available."""
        samples = []
        for i in range(n):
            # Random multi-label
            n_labels = np.random.randint(1, 4)
            label_indices = np.random.choice(len(BEN_LABELS), n_labels, replace=False)
            labels = [BEN_LABELS[j] for j in label_indices]

            samples.append({
                "patch_dir": None,  # Will generate synthetic image
                "labels": labels,
                "text": self._labels_to_text(labels),
            })
        return samples

    @staticmethod
    def _labels_to_text(labels: List[str]) -> str:
        """Convert multi-label list to natural-language description."""
        if not labels:
            return "An aerial image of an unclassified area."
        if len(labels) == 1:
            return f"An aerial image showing {labels[0].lower()}."
        parts = ", ".join(l.lower() for l in labels[:-1])
        return f"An aerial image showing {parts} and {labels[-1].lower()}."

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        if sample["patch_dir"] is not None:
            # Load real image
            patch_dir = Path(sample["patch_dir"])
            # Try to find RGB bands (B04, B03, B02)
            try:
                import rasterio
                bands = []
                for band_name in ["B04", "B03", "B02"]:
                    band_files = list(patch_dir.glob(f"*{band_name}*"))
                    if band_files:
                        with rasterio.open(band_files[0]) as src:
                            bands.append(src.read(1))
                if bands:
                    rgb = np.stack(bands, axis=-1).astype(np.float32)
                    # Percentile normalisation
                    for i in range(3):
                        p2, p98 = np.percentile(rgb[..., i], [2, 98])
                        if p98 > p2:
                            rgb[..., i] = np.clip((rgb[..., i] - p2) / (p98 - p2), 0, 1)
                    rgb = (rgb * 255).astype(np.uint8)
                    image = Image.fromarray(rgb).resize(
                        (self.image_size, self.image_size)
                    )
                else:
                    image = self._synthetic_image()
            except Exception:
                image = self._synthetic_image()
        else:
            image = self._synthetic_image()

        return {
            "image": image,
            "text": sample["text"],
            "labels": sample["labels"],
        }

    def _synthetic_image(self) -> Image.Image:
        """Generate a random synthetic image for testing."""
        arr = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
        return Image.fromarray(arr)


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    """Run LoRA fine-tuning."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # 1. Load model & processor
    logger.info(f"Loading model: {args.model_name}")
    from transformers import (
        Blip2Processor,
        Blip2ForConditionalGeneration,
    )

    processor = Blip2Processor.from_pretrained(args.model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    )

    # 2. Apply LoRA
    from models.adapters import RSLoRAAdapter
    adapter = RSLoRAAdapter(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model = adapter.apply_lora(model)
    model = model.to(device)

    # 3. Load dataset
    dataset = BigEarthNetDataset(
        data_dir=args.data_dir,
        split="train",
        max_samples=args.max_samples,
        image_size=224,
    )

    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        inputs = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # For causal LM training, labels = input_ids
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 4. Optimiser
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # 5. Training loop
    logger.info(f"Starting training: {args.epochs} epochs, {len(dataset)} samples")
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f}")

    # 6. Save LoRA weights
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    RSLoRAAdapter.save_lora(model, str(output_dir))

    # Save training config
    config = vars(args)
    config["device"] = device
    config["final_loss"] = avg_loss
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"✅ Training complete. LoRA weights saved to: {output_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SatQuery AI — LoRA Fine-Tuning on BigEarthNet-MM"
    )
    parser.add_argument(
        "--model_name", type=str, default="Salesforce/blip2-opt-2.7b",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--data_dir", type=Path, default=Path("data/datasets/bigearthnet"),
        help="Path to BigEarthNet dataset.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("checkpoints/lora_bigearthnet"),
        help="Output directory for LoRA weights.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit samples (for debugging).",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
