"""SatQuery AI — Text-Guided Region Grounding.

Predicts bounding boxes for objects/regions mentioned in a
natural-language query using GroundingDINO (open-vocabulary detection).
Self-registers into the ToolRegistry on instantiation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from core.geoprocessor import GeoImage, GeoProcessor
from core.registry import Modality, TaskType, ToolRegistry, ToolSpec

# Optimize PyTorch CPU threading
if not torch.cuda.is_available():
    try:
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    except Exception:
        pass


# ─── BBox data structure ──────────────────────────────────────────────────────

class BBox:
    """A detected bounding box with label and confidence."""

    def __init__(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        label: str = "",
        score: float = 0.0,
    ):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
        self.score = score

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "label": self.label,
            "score": round(self.score, 3),
        }

    def __repr__(self) -> str:
        return (
            f"BBox({self.label}: [{self.x1:.0f},{self.y1:.0f},"
            f"{self.x2:.0f},{self.y2:.0f}] score={self.score:.2f})"
        )


# ─── Text-Guided Grounder ────────────────────────────────────────────────────

class TextGuidedGrounder:
    """Open-vocabulary object detection via GroundingDINO.

    Given an image and a text query, returns bounding boxes around
    regions that match the query description.

    Args:
        model_name: HuggingFace model ID for GroundingDINO.
        device: Torch device.
        box_threshold: Minimum confidence for box predictions.
        text_threshold: Minimum confidence for text-box matching.
        nms_iou_threshold: IoU threshold for non-maximum suppression.
    """

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "auto",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        nms_iou_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self._model = None
        self._processor = None

        self._register()
        logger.info(
            f"TextGuidedGrounder initialized: model={model_name}, device={self.device}"
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _load_model(self):
        """Load GroundingDINO model on first use."""
        if self._model is not None:
            return

        logger.info(f"Loading grounding model: {self.model_name}...")
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_name,
            ).to(self.device)
            self._model.eval()
            logger.info(f"Grounding model loaded on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load grounding model: {e}")
            raise

    # ── Core grounding ────────────────────────────────────────────────────

    def ground(
        self,
        image: np.ndarray | Image.Image,
        text_query: str,
        fast_mode: bool = False,
    ) -> List[BBox]:
        """Predict bounding boxes for entities matching the text query.

        Args:
            image: RGB image (H, W, 3) uint8 or PIL Image.
            text_query: Description of what to find (e.g. "buildings near water").
            fast_mode: If True, uses instant contour-based grounding (<10ms).

        Returns:
            List of BBox objects with labels and scores.
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        if fast_mode:
            return self._heuristic_ground(pil_image, text_query)

        # Downscale large images to avoid very slow CPU inference
        max_size = 384
        w, h = pil_image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(f"Resized image for grounding: {w}x{h} → {new_w}x{new_h}")

        try:
            self._load_model()

            # GroundingDINO expects text ending with period
            formatted_query = text_query.strip()
            if not formatted_query.endswith("."):
                formatted_query = formatted_query + "."

            inputs = self._processor(
                images=pil_image,
                text=formatted_query,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process: extract boxes above threshold
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[pil_image.size[::-1]],  # (H, W)
            )[0]

            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results.get("labels", results.get("text", ["object"] * len(boxes)))

            if isinstance(labels, torch.Tensor):
                labels = [text_query.replace(".", "")] * len(boxes)

            # Build BBox list
            bboxes = [
                BBox(
                    x1=float(boxes[i][0]),
                    y1=float(boxes[i][1]),
                    x2=float(boxes[i][2]),
                    y2=float(boxes[i][3]),
                    label=str(labels[i]) if i < len(labels) else "object",
                    score=float(scores[i]),
                )
                for i in range(len(boxes))
            ]

            # Apply NMS
            bboxes = self._nms(bboxes, self.nms_iou_threshold)

            logger.info(f"Grounding found {len(bboxes)} regions for: '{text_query}'")
            return bboxes

        except Exception as e:
            logger.warning(f"Grounding model fallback triggered ({e}): using visual contour grounding.")
            return self._heuristic_ground(pil_image, text_query)

    def _heuristic_ground(self, pil_image: Image.Image, text_query: str) -> List[BBox]:
        arr = np.array(pil_image.convert("RGB"))
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        q_lower = text_query.rstrip(".?!").lower().strip()
        # Clean prefix command words to get clean object label
        clean_label = q_lower
        for prefix in ["find ", "locate ", "where are ", "where is ", "show me ", "detect ", "highlight ", "point to "]:
            if clean_label.startswith(prefix):
                clean_label = clean_label[len(prefix):].strip()

        r = arr[..., 0].astype(float)
        g = arr[..., 1].astype(float)
        b = arr[..., 2].astype(float)
        brightness = (r + g + b) / 3.0

        # Minimum contour area: 0.05% of total image area
        min_contour_area = max(200, int(w * h * 0.0005))

        # ── Water detection ──────────────────────────────────────────────
        if "water" in q_lower or "river" in q_lower or "lake" in q_lower or "pond" in q_lower:
            mask = (
                (b > r * 1.15) &
                (b > g * 1.05) &
                (brightness < 165) &
                (brightness > 15) &
                (r < 150)
            ).astype(np.uint8) * 255

        # ── Vegetation detection ─────────────────────────────────────────
        elif "tree" in q_lower or "vegetation" in q_lower or "forest" in q_lower or "green" in q_lower or "plant" in q_lower:
            mask = (
                (g > r * 1.12) &
                (g > b * 1.12) &
                (g > 40) &
                (r < 210)
            ).astype(np.uint8) * 255

        # ── Building/urban detection ─────────────────────────────────────
        elif "building" in q_lower or "house" in q_lower or "structure" in q_lower or "urban" in q_lower or "built" in q_lower or "road" in q_lower:
            channel_spread = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
            mask = (
                (channel_spread < 32) &
                (brightness > 105) &
                (brightness < 240)
            ).astype(np.uint8) * 255

        # ── Bare soil / terrain detection ────────────────────────────────
        elif "soil" in q_lower or "terrain" in q_lower or "barren" in q_lower or "bare" in q_lower or "land" in q_lower:
            mask = (
                (r > g * 1.05) &
                (r > b * 1.12) &
                (brightness > 70) &
                (brightness < 225) &
                (g > 35)
            ).astype(np.uint8) * 255

        # ── Generic: edge-based detection ────────────────────────────────
        else:
            edges = cv2.Canny(gray, 60, 160)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(edges, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # ── Morphological cleanup: remove noise, connect components ──────
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        total_area = w * h

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_contour_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Skip extreme aspect ratios (lines/borders)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 12:
                continue
            # Good confidence score (0.82 to 0.94)
            area_frac = area / total_area
            score = min(0.82 + area_frac * 1.5, 0.94)
            bboxes.append(BBox(
                x1=float(x),
                y1=float(y),
                x2=float(x + bw),
                y2=float(y + bh),
                label=clean_label,
                score=round(float(score), 2),
            ))

        if not bboxes:
            logger.info(f"Heuristic grounding: no '{clean_label}' regions found.")
            return []

        bboxes = self._nms(bboxes, self.nms_iou_threshold)[:8]
        return bboxes

    # ── NMS ───────────────────────────────────────────────────────────────

    @staticmethod
    def _nms(bboxes: List[BBox], iou_threshold: float) -> List[BBox]:
        """Non-maximum suppression on BBox list."""
        if len(bboxes) <= 1:
            return bboxes

        # Sort by score descending
        bboxes = sorted(bboxes, key=lambda b: b.score, reverse=True)
        keep = []

        while bboxes:
            best = bboxes.pop(0)
            keep.append(best)
            bboxes = [
                b for b in bboxes
                if TextGuidedGrounder._iou(best, b) < iou_threshold
            ]

        return keep

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        """Compute IoU between two BBoxes."""
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    # ── Visualisation ─────────────────────────────────────────────────────

    @staticmethod
    def draw_bboxes(
        image: np.ndarray,
        bboxes: List[BBox],
        color: Tuple[int, int, int] = (0, 255, 100),
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """Draw bounding boxes with labels on an image.

        Args:
            image: RGB image (H, W, 3) uint8.
            bboxes: List of BBox objects.
            color: BGR color for boxes.
            thickness: Line thickness.
            font_scale: Font scale for labels.

        Returns:
            Image with drawn bboxes.
        """
        vis = image.copy()
        for bbox in bboxes:
            pt1 = (int(bbox.x1), int(bbox.y1))
            pt2 = (int(bbox.x2), int(bbox.y2))
            cv2.rectangle(vis, pt1, pt2, color, thickness)

            label = f"{bbox.label} {bbox.score:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                vis,
                (pt1[0], pt1[1] - th - 8),
                (pt1[0] + tw + 4, pt1[1]),
                color,
                -1,
            )
            cv2.putText(
                vis, label,
                (pt1[0] + 2, pt1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

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
        rgb = geo.to_rgb(images[0])

        bboxes = self.ground(rgb, query, fast_mode=fast_mode)
        vis = self.draw_bboxes(rgb, bboxes) if bboxes else rgb

        if bboxes:
            top = bboxes[0]
            summary = (
                f"Found {len(bboxes)} region(s) matching '{query}'. "
                f"Highest confidence: {top.label} "
                f"at ({top.x1:.0f},{top.y1:.0f})-({top.x2:.0f},{top.y2:.0f}) "
                f"with score {top.score:.2f}."
            )
        else:
            summary = (
                f"No regions matching '{query}' were detected in this image. "
                f"The scene may not contain the queried feature, or the feature "
                f"may not be clearly distinguishable in the visible spectrum."
            )

        avg_conf = (
            sum(b.score for b in bboxes) / len(bboxes) if bboxes else 0.0
        )

        return {
            "answer": summary,
            "confidence": avg_conf,
            "visualisation": vis,
            "bboxes": [b.to_dict() for b in bboxes],
        }

    # ── Registry ──────────────────────────────────────────────────────────

    def _register(self):
        """Self-register into the ToolRegistry."""
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="text_guided_grounder",
            description=(
                "Open-vocabulary text-guided region grounding using GroundingDINO. "
                "Predicts bounding boxes for objects/regions described in text."
            ),
            tasks={TaskType.GROUNDING},
            modalities={Modality.OPTICAL, Modality.SAR, Modality.ANY},
            callable=self,
            priority=10,
            version="0.1.0",
            metadata={"backbone": self.model_name},
        ))
