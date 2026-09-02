"""SatQuery AI — Optical + SAR Joint Analysis & Fusion.

Combines optical spectral indices (NDVI, NDWI) with SAR backscatter
features (VV, VH, VV/VH ratio) for land-cover classification and
flood/water body detection.
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


# ─── Fusion Classifier Network ───────────────────────────────────────────────

class FusionClassifierCNN(nn.Module):
    """Lightweight CNN for pixel-wise land-cover classification
    from concatenated optical+SAR feature stacks.

    Input channels: optical features + SAR features (varies by config).
    Output: num_classes probability maps.

    Args:
        in_channels: Number of input feature channels.
        hidden_channels: List of hidden layer channel counts.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: Optional[List[int]] = None,
        num_classes: int = 5,
    ):
        super().__init__()
        hidden_channels = hidden_channels or [64, 128, 64]

        layers = []
        ch_in = in_channels
        for ch_out in hidden_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out

        layers.append(nn.Conv2d(ch_in, num_classes, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, C_in, H, W) → (B, num_classes, H, W)."""
        return self.network(x)


# ─── Optical-SAR Fusion Tool ─────────────────────────────────────────────────

class OpticalSARFusion:
    """Optical + SAR joint analysis and feature fusion.

    Pipeline:
        1. Extract optical features: RGB channels + spectral indices (NDVI, NDWI)
        2. Extract SAR features: VV, VH bands + VV/VH ratio
        3. Concatenate features → feed to FusionClassifierCNN
        4. Produce pixel-wise classification map
        5. Optionally feed results to VLM for natural-language analysis

    Args:
        device: Torch device.
        num_classes: Number of land-cover classes.
        class_labels: Human-readable labels for each class.
    """

    DEFAULT_LABELS = ["Water", "Built-Up", "Vegetation", "Flood", "Other"]

    # Colors for classification map visualisation (RGB)
    CLASS_COLORS = [
        (30, 144, 255),   # Water — Dodger Blue
        (220, 80, 60),    # Built-Up — Crimson
        (34, 180, 34),    # Vegetation — Green
        (255, 165, 0),    # Flood — Orange
        (128, 128, 128),  # Other — Gray
    ]

    def __init__(
        self,
        device: str = "auto",
        num_classes: int = 5,
        class_labels: Optional[List[str]] = None,
    ):
        self.device = self._resolve_device(device)
        self.num_classes = num_classes
        self.class_labels = class_labels or self.DEFAULT_LABELS
        self._classifier = None
        self._vqa = None

        self._register()
        logger.info(
            f"OpticalSARFusion initialized: {num_classes} classes, device={self.device}"
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _load_classifier(self, in_channels: int):
        if self._classifier is not None:
            return
        logger.info("Initialising Fusion Classifier CNN...")
        self._classifier = FusionClassifierCNN(
            in_channels=in_channels,
            num_classes=self.num_classes,
        ).to(self.device).eval()
        logger.info("Fusion classifier loaded.")

    def _load_vqa(self):
        if self._vqa is not None:
            return
        from models.vqa_caption import RSVQACaptioner
        self._vqa = RSVQACaptioner.__new__(RSVQACaptioner)
        self._vqa.model_name = "Salesforce/blip2-opt-2.7b"
        self._vqa.max_new_tokens = 64 if self.device == "cpu" else 256
        self._vqa.temperature = 0.7
        self._vqa.device = self.device
        self._vqa._model = None
        self._vqa._processor = None
        self._vqa._lora_path = None

    # ── Feature Extraction ────────────────────────────────────────────────

    @staticmethod
    def extract_optical_features(geo_image: GeoImage) -> np.ndarray:
        """Extract optical features: RGB + spectral indices.

        Returns:
            Feature stack of shape (C, H, W) where C = 3 (RGB) + 2 (NDVI, NDWI) = 5
            for 4+ band imagery, or C = 3 for RGB-only.
        """
        array = geo_image.array  # (C, H, W) float32 [0, 1]

        # Always include RGB (first 3 bands)
        rgb = array[:min(3, array.shape[0])]

        features = [rgb]

        # Add spectral indices if NIR band available (band 4+)
        if array.shape[0] >= 4:
            geo = GeoProcessor()
            ndvi = GeoProcessor.compute_ndvi(geo_image)  # (H, W)
            ndwi = GeoProcessor.compute_ndwi(geo_image)  # (H, W)
            features.append(ndvi[np.newaxis])
            features.append(ndwi[np.newaxis])

        return np.concatenate(features, axis=0).astype(np.float32)

    @staticmethod
    def extract_sar_features(geo_image: GeoImage) -> np.ndarray:
        """Extract SAR features: VV, VH, VV/VH ratio.

        Returns:
            Feature stack of shape (3, H, W) or (C, H, W) for C-band data.
        """
        array = geo_image.array  # (C, H, W)

        if array.shape[0] >= 2:
            vv = array[0]  # First band = VV
            vh = array[1]  # Second band = VH

            # VV/VH ratio (useful for water/flood detection)
            epsilon = 1e-10
            ratio = vv / (vh + epsilon)
            # Normalise ratio to [0, 1]
            ratio = np.clip(ratio / ratio.max(), 0, 1) if ratio.max() > 0 else ratio

            return np.stack([vv, vh, ratio], axis=0).astype(np.float32)
        else:
            # Single band SAR — duplicate for 3 channels
            band = array[0]
            return np.stack([band, band, band], axis=0).astype(np.float32)

    # ── Classification ────────────────────────────────────────────────────

    def fuse_and_classify(
        self,
        optical: GeoImage,
        sar: GeoImage,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Fuse optical and SAR features and classify each pixel.

        Args:
            optical: Optical GeoImage.
            sar: SAR GeoImage.

        Returns:
            (class_map, class_distribution)
            class_map: int array (H, W) with class indices.
            class_distribution: dict mapping class labels to percentage.
        """
        opt_feat = self.extract_optical_features(optical)  # (C1, H, W)
        sar_feat = self.extract_sar_features(sar)  # (C2, H, W)

        # Resize SAR features to match optical if needed
        h, w = opt_feat.shape[1], opt_feat.shape[2]
        if sar_feat.shape[1] != h or sar_feat.shape[2] != w:
            sar_resized = np.zeros((sar_feat.shape[0], h, w), dtype=np.float32)
            for i in range(sar_feat.shape[0]):
                sar_resized[i] = cv2.resize(sar_feat[i], (w, h))
            sar_feat = sar_resized

        # Concatenate
        fused = np.concatenate([opt_feat, sar_feat], axis=0)  # (C1+C2, H, W)
        in_channels = fused.shape[0]

        # Downscale large feature stacks for faster CNN inference
        max_size = 512
        orig_h, orig_w = h, w
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            fused_resized = np.zeros((in_channels, new_h, new_w), dtype=np.float32)
            for i in range(in_channels):
                fused_resized[i] = cv2.resize(fused[i], (new_w, new_h))
            fused = fused_resized
            logger.debug(f"Resized fused features: {orig_w}x{orig_h} → {new_w}x{new_h}")

        self._load_classifier(in_channels)

        # To tensor
        x = torch.from_numpy(fused).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._classifier(x)  # (1, num_classes, H, W)
            # Resize logits back to original spatial size
            if max(orig_h, orig_w) > max_size:
                logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)
            class_map = probs.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)

        # Compute class distribution
        total_pixels = class_map.size
        distribution = {}
        for i, label in enumerate(self.class_labels):
            count = np.sum(class_map == i)
            distribution[label] = round(count / total_pixels * 100, 1)

        logger.info(f"Classification distribution: {distribution}")
        return class_map, distribution

    # ── Visualisation ─────────────────────────────────────────────────────

    def colorize_class_map(self, class_map: np.ndarray) -> np.ndarray:
        """Convert integer class map to RGB colour image.

        Args:
            class_map: int array (H, W) with class indices.

        Returns:
            RGB image (H, W, 3) uint8.
        """
        h, w = class_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for i, color in enumerate(self.CLASS_COLORS):
            rgb[class_map == i] = color

        return rgb

    def create_legend_image(self, height: int = 200, width: int = 250) -> np.ndarray:
        """Create a colour legend image.

        Returns:
            RGB legend image (height, width, 3) uint8.
        """
        legend = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark background
        row_h = height // (len(self.class_labels) + 1)

        for i, (label, color) in enumerate(zip(self.class_labels, self.CLASS_COLORS)):
            y = (i + 1) * row_h
            # Color swatch
            cv2.rectangle(legend, (10, y - 12), (30, y + 4), color, -1)
            # Label text
            cv2.putText(
                legend, label,
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

        return legend

    # ── Analysis with VLM ─────────────────────────────────────────────────

    def analyze(
        self,
        optical: GeoImage,
        sar: GeoImage,
        query: str,
        fast_mode: bool = False,
    ) -> Tuple[str, np.ndarray, float, np.ndarray, Dict[str, float]]:
        """Full analysis: classify + VLM reasoning.

        Args:
            optical: Optical GeoImage.
            sar: SAR GeoImage.
            query: Natural-language query about the scene.
            fast_mode: If True, uses instant analytical feature heuristics.

        Returns:
            (answer_text, colorised_map, confidence, class_map, distribution)
        """
        # Classification
        class_map, distribution = self.fuse_and_classify(optical, sar)
        color_map = self.colorize_class_map(class_map)

        # Build context description from classification
        dist_str = ", ".join(f"{k}: {v}%" for k, v in distribution.items() if v > 0)
        context = (
            f"This remote sensing scene has been classified using optical and SAR "
            f"data fusion. Land cover distribution: {dist_str}. "
        )

        # Use VLM for natural-language answer
        self._load_vqa()
        geo = GeoProcessor()
        optical_rgb = geo.to_rgb(optical)

        full_query = f"{context} {query}"
        answer, confidence = self._vqa.answer(optical_rgb, full_query, fast_mode=fast_mode)

        # Enrich answer with classification stats
        answer = f"{answer}\n\nLand Cover Breakdown: {dist_str}"

        return answer, color_map, confidence, class_map, distribution

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
        # Determine which image is optical vs SAR
        optical, sar = self._identify_images(images)

        answer, color_map, confidence, class_map, distribution = self.analyze(
            optical, sar, query, fast_mode=fast_mode
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "visualisation": color_map,
            "class_map": class_map,
        }

    @staticmethod
    def _identify_images(
        images: List[GeoImage],
    ) -> Tuple[GeoImage, GeoImage]:
        """Identify which image is optical and which is SAR.

        Heuristic: SAR images typically have fewer bands (1-2)
        and may have VV/VH band names.
        """
        if len(images) < 2:
            raise ValueError("Cross-modal fusion requires 2 images (optical + SAR).")

        img_a, img_b = images[0], images[1]

        sar_indicators = {"vv", "vh", "hh", "hv", "backscatter", "sigma"}
        a_bands = {b.lower() for b in img_a.band_names}
        b_bands = {b.lower() for b in img_b.band_names}

        a_is_sar = bool(a_bands & sar_indicators) or img_a.num_bands <= 2
        b_is_sar = bool(b_bands & sar_indicators) or img_b.num_bands <= 2

        if a_is_sar and not b_is_sar:
            return img_b, img_a  # optical, sar
        elif b_is_sar and not a_is_sar:
            return img_a, img_b  # optical, sar
        else:
            # Default: first = optical, second = sar
            logger.warning(
                "Could not determine optical vs SAR from band names. "
                "Assuming first image is optical, second is SAR."
            )
            return img_a, img_b

    # ── Registry ──────────────────────────────────────────────────────────

    def _register(self):
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="optical_sar_fusion",
            description=(
                "Cross-modal optical+SAR joint analysis and fusion. "
                "Combines spectral indices (NDVI, NDWI) with SAR backscatter "
                "(VV/VH) for land-cover classification and flood detection."
            ),
            tasks={TaskType.CROSS_MODAL_FUSION},
            modalities={Modality.CROSS_MODAL, Modality.ANY},
            callable=self,
            priority=10,
            version="0.1.0",
        ))
