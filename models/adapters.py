"""SatQuery AI — Remote-Sensing Adaptation Layers.

Provides LoRA adapters and band-projection modules for adapting
pretrained vision-language models to multi-band RS imagery.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


# ─── Band Projection ─────────────────────────────────────────────────────────

class BandProjection(nn.Module):
    """Projects N-band remote-sensing imagery to 3-channel RGB.

    Pretrained VLMs expect 3-channel (RGB) input. This lightweight
    convolutional adapter maps arbitrary band counts to 3 channels
    while preserving spatial resolution.

    Args:
        in_channels: Number of input spectral bands.
        out_channels: Number of output channels (default 3 for RGB).
        init_weights: If True, initialise weights for common band combos.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        init_weights: bool = True,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=True),
        )

        if init_weights:
            self._init_weights(in_channels, out_channels)

    def _init_weights(self, in_ch: int, out_ch: int):
        """Smart init: first 3 bands get identity-like mapping."""
        with torch.no_grad():
            # First conv: let first 3 input channels pass through
            w = self.projection[0].weight  # (out_ch*2, in_ch, 1, 1)
            nn.init.kaiming_normal_(w)
            # Boost first 3 channels
            n = min(in_ch, out_ch)
            for i in range(n):
                w[i, i, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, C_in, H, W) → (B, 3, H, W)."""
        return self.projection(x)


# ─── LoRA Adapter ─────────────────────────────────────────────────────────────

class RSLoRAAdapter:
    """Wrapper for applying PEFT LoRA to HuggingFace models.

    Provides a clean API for:
    - Applying LoRA config to a model
    - Saving/loading LoRA weights
    - Merging LoRA weights into the base model

    Args:
        rank: LoRA rank (dimensionality of low-rank matrices).
        alpha: LoRA scaling factor.
        target_modules: List of module name patterns to apply LoRA to.
        dropout: LoRA dropout rate.
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: int = 32,
        target_modules: Optional[list[str]] = None,
        dropout: float = 0.05,
    ):
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.dropout = dropout

    def apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA adapters to a HuggingFace model.

        Returns the PEFT-wrapped model.
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType as PeftTask
        except ImportError:
            logger.warning(
                "peft not installed — LoRA not applied. "
                "Install with: pip install peft"
            )
            return model

        config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias="none",
            task_type=PeftTask.CAUSAL_LM,
        )

        peft_model = get_peft_model(model, config)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        logger.info(
            f"LoRA applied: rank={self.rank}, alpha={self.alpha} | "
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )
        return peft_model

    @staticmethod
    def load_pretrained_lora(model: nn.Module, path: str) -> nn.Module:
        """Load pre-trained LoRA weights from a checkpoint directory.

        Args:
            model: The base HuggingFace model.
            path: Path to the LoRA checkpoint directory.

        Returns:
            Model with LoRA weights loaded.
        """
        try:
            from peft import PeftModel
        except ImportError:
            logger.error("peft not installed — cannot load LoRA weights.")
            return model

        peft_model = PeftModel.from_pretrained(model, path)
        logger.info(f"Loaded LoRA weights from: {path}")
        return peft_model

    @staticmethod
    def merge_and_unload(model: nn.Module) -> nn.Module:
        """Merge LoRA weights into the base model and remove adapters.

        Useful for inference optimisation — eliminates LoRA overhead.
        """
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
            logger.info("LoRA weights merged into base model.")
        return model

    @staticmethod
    def save_lora(model: nn.Module, path: str) -> None:
        """Save only the LoRA adapter weights."""
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(path)
            logger.info(f"LoRA weights saved to: {path}")
        else:
            logger.warning("Model is not a PEFT model — cannot save LoRA weights.")
