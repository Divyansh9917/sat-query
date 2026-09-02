"""SatQuery AI — Bi-temporal Change Detection & Change VQA.

Combines a siamese CNN difference network for spatial change maps
with VLM-based change captioning and change-VQA.
Self-registers into the ToolRegistry on instantiation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from core.geoprocessor import GeoImage, GeoProcessor
from core.registry import Modality, TaskType, ToolRegistry, ToolSpec


# ─── Siamese Feature Extractor ────────────────────────────────────────────────

class SiameseChangeNet(nn.Module):
    """Siamese network for bi-temporal change detection.

    Architecture:
        - Shared ResNet-18 backbone (up to layer3) for feature extraction
        - L2 difference of features
        - Lightweight decoder to produce binary change mask

    Args:
        pretrained: Use ImageNet pretrained backbone.
        threshold: Activation threshold for binary mask.
    """

    def __init__(self, pretrained: bool = True, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

        # Shared encoder (ResNet-18 features)
        import torchvision.models as models
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        # Decoder: upscale difference features to mask
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        img_pre: torch.Tensor,
        img_post: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            img_pre: Pre-change image (B, 3, H, W).
            img_post: Post-change image (B, 3, H, W).

        Returns:
            (change_prob_map, binary_change_mask) both shape (B, 1, H', W')
            where H', W' may differ from input due to downsampling.
        """
        feat_pre = self.encoder(img_pre)
        feat_post = self.encoder(img_post)

        # L2 difference
        diff = torch.abs(feat_pre - feat_post)

        # Decode to change probability
        change_prob = self.decoder(diff)

        # Binary mask
        change_mask = (change_prob > self.threshold).float()

        return change_prob, change_mask


# ─── Change Detector Tool ─────────────────────────────────────────────────────

class ChangeDetector:
    """Bi-temporal change detection combining spatial masks with VLM reasoning.

    Capabilities:
        - detect_change(pre, post) → binary change mask
        - describe_change(pre, post, query) → natural-language change description
        - Supports Change-VQA: answer questions about what changed

    Args:
        device: Torch device.
        change_threshold: Threshold for binarising change probability.
        vqa_model_name: VLM backbone for change captioning/VQA.
    """

    def __init__(
        self,
        device: str = "auto",
        change_threshold: float = 0.5,
        vqa_model_name: str = "Salesforce/blip2-opt-2.7b",
    ):
        self.device = self._resolve_device(device)
        self.change_threshold = change_threshold
        self.vqa_model_name = vqa_model_name
        self._siamese = None
        self._vqa = None

        self._register()
        logger.info(f"ChangeDetector initialized: device={self.device}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _load_siamese(self):
        if self._siamese is not None:
            return
        logger.info("Loading Siamese change detection network...")
        self._siamese = SiameseChangeNet(
            pretrained=True,
            threshold=self.change_threshold,
        ).to(self.device).eval()
        logger.info("Siamese network loaded.")

    def _load_vqa(self):
        """Reuse the RSVQACaptioner for change description."""
        if self._vqa is not None:
            return
        from models.vqa_caption import RSVQACaptioner
        # Don't re-register — the VQA model manages its own registration
        self._vqa = RSVQACaptioner.__new__(RSVQACaptioner)
        self._vqa.model_name = self.vqa_model_name
        self._vqa.max_new_tokens = 64 if self.device == "cpu" else 256
        self._vqa.temperature = 0.7
        self._vqa.device = self.device
        self._vqa._model = None
        self._vqa._processor = None
        self._vqa._lora_path = None

    # ── Image preparation ─────────────────────────────────────────────────

    def _prepare_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        """Convert RGB uint8 (H,W,3) to normalised tensor (1,3,H,W)."""
        img = rgb.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)

    # ── Core Methods ──────────────────────────────────────────────────────

    def detect_change(
        self,
        img_pre: np.ndarray,
        img_post: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate spatial change map between two images.

        Args:
            img_pre: Pre-change RGB image (H, W, 3) uint8.
            img_post: Post-change RGB image (H, W, 3) uint8.

        Returns:
            (change_prob_map, binary_mask) both numpy arrays.
            change_prob_map: float32 (H, W) in [0, 1].
            binary_mask: uint8 (H, W) in {0, 255}.
        """
        self._load_siamese()

        orig_h, orig_w = img_pre.shape[:2]

        # Resize to match if needed
        if img_post.shape[:2] != (orig_h, orig_w):
            img_post = cv2.resize(img_post, (orig_w, orig_h))

        # Downscale large images for faster inference
        max_size = 512
        h, w = orig_h, orig_w
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_pre_rs = cv2.resize(img_pre, (new_w, new_h))
            img_post_rs = cv2.resize(img_post, (new_w, new_h))
            logger.debug(f"Resized images for change detection: {w}x{h} → {new_w}x{new_h}")
        else:
            img_pre_rs = img_pre
            img_post_rs = img_post
            new_h, new_w = h, w

        t_pre = self._prepare_tensor(img_pre_rs)
        t_post = self._prepare_tensor(img_post_rs)

        with torch.no_grad():
            prob_map, mask = self._siamese(t_pre, t_post)

        # Resize back to original resolution
        prob_map = F.interpolate(
            prob_map, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        mask = F.interpolate(
            mask, size=(orig_h, orig_w), mode="nearest"
        )

        prob_np = prob_map.squeeze().cpu().numpy()
        mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Compute change percentage
        change_pct = np.mean(mask_np > 0) * 100
        logger.info(f"Change detected: {change_pct:.1f}% of area changed.")

        return prob_np, mask_np

    def describe_change(
        self,
        img_pre: np.ndarray,
        img_post: np.ndarray,
        query: str = "Describe the changes between these two images.",
        fast_mode: bool = False,
    ) -> Tuple[str, float]:
        """Describe changes between two images using VLM or fast analytics.

        Creates a side-by-side comparison image and feeds it to the VLM.

        Args:
            img_pre: Pre-change RGB image (H, W, 3) uint8.
            img_post: Post-change RGB image (H, W, 3) uint8.
            query: Question about the changes.
            fast_mode: If True, uses instant analytical feature heuristics.

        Returns:
            (description_text, confidence)
        """
        self._load_vqa()

        # Create side-by-side comparison
        h, w = img_pre.shape[:2]
        if img_post.shape[:2] != (h, w):
            img_post = cv2.resize(img_post, (w, h))

        # Add labels
        comparison = np.hstack([img_pre, img_post])

        # Prepend context to query
        context_query = (
            f"This image shows a before (left) and after (right) comparison "
            f"of a remote sensing scene. {query}"
        )

        answer, confidence = self._vqa.answer(comparison, context_query, fast_mode=fast_mode)
        return answer, confidence

    # ── Visualisation ─────────────────────────────────────────────────────

    @staticmethod
    def overlay_change_mask(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 50, 50),
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay a change mask on an image with transparency.

        Args:
            image: RGB image (H, W, 3) uint8.
            mask: Binary mask (H, W) uint8 {0, 255}.
            color: RGB color for changed regions.
            alpha: Transparency of the overlay.

        Returns:
            Image with overlay (H, W, 3) uint8.
        """
        vis = image.copy()
        mask_bool = mask > 0

        overlay = np.zeros_like(vis)
        overlay[mask_bool] = color

        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
        # Keep non-masked areas unchanged
        vis[~mask_bool] = image[~mask_bool]

        return vis

    # ── Tool Interface ────────────────────────────────────────────────────

    def __call__(
        self,
        images: List[GeoImage],
        query: str,
        task: TaskType,
        fast_mode: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Unified tool interface called by the orchestrator."""
        geo = GeoProcessor()
        rgb_pre = geo.to_rgb(images[0])
        rgb_post = geo.to_rgb(images[1])

        if task == TaskType.CHANGE_DETECTION:
            prob_map, mask = self.detect_change(rgb_pre, rgb_post)
            vis = self.overlay_change_mask(rgb_post, mask)

            change_pct = np.mean(mask > 0) * 100
            answer = (
                f"Change detection complete. "
                f"Approximately {change_pct:.1f}% of the scene area has changed "
                f"between the two temporal images."
            )

            return {
                "answer": answer,
                "confidence": 0.8,
                "visualisation": vis,
                "change_mask": mask,
            }

        elif task == TaskType.CHANGE_CAPTION:
            # Generate both mask and description
            _, mask = self.detect_change(rgb_pre, rgb_post)
            description, confidence = self.describe_change(rgb_pre, rgb_post, query, fast_mode=fast_mode)
            vis = self.overlay_change_mask(rgb_post, mask)

            return {
                "answer": description,
                "confidence": confidence,
                "visualisation": vis,
                "change_mask": mask,
            }

        else:
            # Fallback: change detection
            _, mask = self.detect_change(rgb_pre, rgb_post)
            vis = self.overlay_change_mask(rgb_post, mask)
            return {
                "answer": "Change map generated.",
                "confidence": 0.7,
                "visualisation": vis,
                "change_mask": mask,
            }

    # ── Registry ──────────────────────────────────────────────────────────

    def _register(self):
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="change_detector",
            description=(
                "Bi-temporal change detection using siamese feature differencing "
                "and VLM-based change captioning/VQA."
            ),
            tasks={TaskType.CHANGE_DETECTION, TaskType.CHANGE_CAPTION},
            modalities={Modality.BITEMPORAL, Modality.ANY},
            callable=self,
            priority=10,
            version="0.1.0",
        ))
