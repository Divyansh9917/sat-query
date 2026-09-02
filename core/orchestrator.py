"""SatQuery AI — Agentic Orchestrator.

Classifies user intent, validates inputs, dispatches to specialist tools,
and produces an auditable execution trace for every query.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from loguru import logger

from core.geoprocessor import GeoImage, GeoProcessor
from core.registry import Modality, TaskType, ToolRegistry


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ExecutionTrace:
    """Auditable record of a single orchestrator run."""
    task: str
    selected_model: str
    inputs_summary: str
    confidence: float
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "success"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    """Complete response from the orchestrator."""
    answer: str
    confidence: float
    visualisation: Optional[np.ndarray] = None  # RGB overlay image (H, W, 3) uint8
    change_mask: Optional[np.ndarray] = None  # Binary mask (H, W) for change detection
    class_map: Optional[np.ndarray] = None  # Classification map (H, W) int
    bboxes: Optional[List[Dict[str, Any]]] = None  # Grounding bounding boxes
    trace: Optional[ExecutionTrace] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-safe dict (without numpy arrays)."""
        d = {
            "answer": self.answer,
            "confidence": self.confidence,
            "bboxes": self.bboxes,
        }
        if self.trace:
            d["trace"] = self.trace.to_dict()
        return d


# ─── Intent Classification ───────────────────────────────────────────────────

class InputConfig(str, Enum):
    """Validated input configurations."""
    SINGLE_OPTICAL = "single_optical"
    SINGLE_SAR = "single_sar"
    BITEMPORAL_PAIR = "bitemporal_pair"
    CROSS_MODAL_PAIR = "cross_modal_pair"


# Keyword patterns for rule-based intent classification.
# Each pattern has an associated weight. Higher weights = stronger signal.
# Patterns are (regex, weight) tuples.
_INTENT_PATTERNS: Dict[TaskType, List[tuple]] = {
    TaskType.CHANGE_CAPTION: [
        # Must be checked BEFORE generic caption — these contain both
        # "describe/explain" AND "change" keywords.
        (r"\b(describe|explain|summarize)\b.*\bchange", 3),
        (r"\bchange\b.*\b(describe|explain|summarize)\b", 3),
        (r"\bwhat has changed\b", 3),
        (r"\bhow has .* changed\b", 3),
        (r"\bchange description\b", 3),
    ],
    TaskType.CHANGE_DETECTION: [
        (r"\bdetect.* changes?\b", 3),
        (r"\bchanges?.* detect\b", 3),
        (r"\bchange map\b", 3),
        (r"\bchange mask\b", 3),
        (r"\bwhat.* different\b", 2),
        (r"\bbefore and after\b", 2),
        (r"\b(changes?|changed|difference|differ)\b", 2),
    ],
    TaskType.CROSS_MODAL_FUSION: [
        (r"\b(sar|radar)\b", 3),
        (r"\boptical.* sar|sar.* optical\b", 3),
        (r"\b(fusion|fuse)\b", 2),
        (r"\b(flood|water body|built.?up|backscatter|vv|vh)\b", 2),
    ],
    TaskType.GROUNDING: [
        (r"\b(where|locate|find|show me|point to|highlight|mark)\b", 3),
        (r"\b(bbox|bounding box)\b", 3),
        (r"\b(region|area|position|location of)\b", 2),
    ],
    TaskType.CAPTION: [
        (r"\b(describe|caption|summarize|summary|explain|tell me about|overview)\b", 2),
        (r"\b(what do you see|what is shown|what does .* show)\b", 2),
    ],
    TaskType.VQA: [
        (r"\b(how many|count|number of)\b", 2),
        (r"\b(is there|are there|what is|what are|which)\b", 2),
        (r"\b(identify|recognize|classify|what type|what kind)\b", 2),
        (r"\?\s*$", 1),  # Question mark — weak signal, tiebreaker only
    ],
}

# Priority ordering: when scores are tied, prefer more-specific intents
_INTENT_PRIORITY: Dict[TaskType, int] = {
    TaskType.CHANGE_CAPTION: 60,
    TaskType.CHANGE_DETECTION: 50,
    TaskType.CROSS_MODAL_FUSION: 40,
    TaskType.GROUNDING: 30,
    TaskType.CAPTION: 20,
    TaskType.VQA: 10,
}


class Orchestrator:
    """Agentic controller that routes queries to specialist tools.

    Workflow:
        1. classify_intent(query) → TaskType
        2. detect_modality(images) → Modality
        3. validate_inputs(task, images) → InputConfig
        4. dispatch(task, images, query) → AgentResponse
    """

    def __init__(self, confidence_threshold: float = 0.3):
        self.registry = ToolRegistry()
        self.geo = GeoProcessor()
        self.confidence_threshold = confidence_threshold
        logger.info("Orchestrator initialized.")

    # ── Intent Classification ─────────────────────────────────────────────

    def classify_intent(self, query: str) -> Tuple[TaskType, float]:
        """Classify user intent from a natural-language query.

        Uses weighted keyword/regex rules with priority-based tiebreaking.
        Returns (task_type, confidence).
        """
        query_lower = query.lower().strip()
        scores: Dict[TaskType, float] = {task: 0.0 for task in TaskType}

        for task, patterns in _INTENT_PATTERNS.items():
            for pattern, weight in patterns:
                if re.search(pattern, query_lower):
                    scores[task] += weight

        # Find best match with priority tiebreaking
        best_task = max(
            scores,
            key=lambda t: (scores[t], _INTENT_PRIORITY.get(t, 0)),
        )
        best_score = scores[best_task]
        total = sum(scores.values())

        if best_score == 0:
            # Default to VQA for general questions
            logger.info(f"No intent matched for query — defaulting to VQA.")
            return TaskType.VQA, 0.5

        confidence = min(best_score / max(total, 1), 1.0)
        # Boost confidence if there's a clear winner
        sorted_scores = sorted(scores.values(), reverse=True)
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0
        if best_score > runner_up:
            confidence = min(confidence + 0.2, 1.0)

        logger.info(
            f"Intent classified: {best_task.value} "
            f"(confidence={confidence:.2f}, query='{query[:60]}...')"
        )
        return best_task, confidence

    # ── Modality Detection ────────────────────────────────────────────────

    @staticmethod
    def detect_modality(images: List[GeoImage]) -> Modality:
        """Detect the input modality based on image count and band structure.

        Heuristics:
            - Single image with ≤ 4 bands → OPTICAL
            - Single image with SAR-like band names or 1-2 bands → SAR
            - Two images → BITEMPORAL (same sensor) or CROSS_MODAL (different)
        """
        if len(images) == 0:
            return Modality.ANY

        if len(images) == 1:
            img = images[0]
            sar_indicators = {"vv", "vh", "hh", "hv", "backscatter", "sigma"}
            band_lower = {b.lower() for b in img.band_names}
            if band_lower & sar_indicators or img.num_bands <= 2:
                return Modality.SAR
            return Modality.OPTICAL

        if len(images) == 2:
            img_a, img_b = images
            bands_a = {b.lower() for b in img_a.band_names}
            bands_b = {b.lower() for b in img_b.band_names}
            sar_indicators = {"vv", "vh", "hh", "hv", "backscatter", "sigma"}

            a_is_sar = bool(bands_a & sar_indicators) or img_a.num_bands <= 2
            b_is_sar = bool(bands_b & sar_indicators) or img_b.num_bands <= 2

            if a_is_sar != b_is_sar:
                return Modality.CROSS_MODAL
            return Modality.BITEMPORAL

        return Modality.ANY

    # ── Input Validation ──────────────────────────────────────────────────

    @staticmethod
    def validate_inputs(
        task: TaskType,
        images: List[GeoImage],
    ) -> InputConfig:
        """Validate that the input image configuration matches the task.

        Raises ValueError with a descriptive message if validation fails.
        """
        n = len(images)

        # Tasks requiring exactly one image
        single_tasks = {TaskType.VQA, TaskType.CAPTION, TaskType.GROUNDING}
        # Tasks requiring exactly two images
        pair_tasks = {TaskType.CHANGE_DETECTION, TaskType.CHANGE_CAPTION}
        # Tasks requiring cross-modal pair
        fusion_tasks = {TaskType.CROSS_MODAL_FUSION}

        if task in single_tasks:
            if n < 1:
                raise ValueError(
                    f"Task '{task.value}' requires at least 1 image, got {n}."
                )
            # Use first image if multiple provided
            img = images[0]
            sar_indicators = {"vv", "vh", "hh", "hv"}
            if {b.lower() for b in img.band_names} & sar_indicators:
                return InputConfig.SINGLE_SAR
            return InputConfig.SINGLE_OPTICAL

        if task in pair_tasks:
            if n < 2:
                raise ValueError(
                    f"Task '{task.value}' requires 2 bi-temporal images, got {n}."
                )
            return InputConfig.BITEMPORAL_PAIR

        if task in fusion_tasks:
            if n < 2:
                raise ValueError(
                    f"Task '{task.value}' requires both an Optical and SAR image, got {n}."
                )
            return InputConfig.CROSS_MODAL_PAIR

        return InputConfig.SINGLE_OPTICAL

    # ── Dispatch ──────────────────────────────────────────────────────────

    def dispatch(
        self,
        images: List[GeoImage],
        query: str,
        fast_mode: bool = True,
    ) -> AgentResponse:
        """Main entry point: classify, validate, route, and trace.

        Args:
            images: List of loaded GeoImages.
            query: Natural-language user query.
            fast_mode: If True, uses ultra-fast analytical spectral heuristics and optimized inference.

        Returns:
            AgentResponse with answer, visualisation, and execution trace.
        """
        t_start = time.perf_counter()

        # 1. Classify intent
        task, intent_conf = self.classify_intent(query)

        # 2. Detect modality
        modality = self.detect_modality(images)

        # 3. Validate inputs
        try:
            input_config = self.validate_inputs(task, images)
        except ValueError as e:
            trace = ExecutionTrace(
                task=task.value,
                selected_model="none",
                inputs_summary=f"{len(images)} images",
                confidence=0.0,
                latency_ms=(time.perf_counter() - t_start) * 1000,
                status="error",
                error=str(e),
            )
            return AgentResponse(
                answer=f"Input validation failed: {e}",
                confidence=0.0,
                trace=trace,
            )

        # 4. Find best tool
        tool_spec = self.registry.best_match(task, modality)
        if tool_spec is None:
            # Fallback: try with ANY modality
            tool_spec = self.registry.best_match(task, Modality.ANY)

        if tool_spec is None:
            trace = ExecutionTrace(
                task=task.value,
                selected_model="none",
                inputs_summary=f"{len(images)} images, config={input_config.value}",
                confidence=intent_conf,
                latency_ms=(time.perf_counter() - t_start) * 1000,
                status="error",
                error=f"No tool registered for task={task.value}, modality={modality.value}",
            )
            return AgentResponse(
                answer=(
                    f"No specialist model available for task '{task.value}' "
                    f"with modality '{modality.value}'. "
                    f"Please ensure models are loaded."
                ),
                confidence=0.0,
                trace=trace,
            )

        # 5. Execute the tool
        mode_label = "Fast Mode" if fast_mode else "Neural Mode"
        logger.info(
            f"Dispatching: task={task.value}, model={tool_spec.name}, "
            f"modality={modality.value}, config={input_config.value}, mode={mode_label}"
        )

        try:
            result = tool_spec.callable(
                images=images,
                query=query,
                task=task,
                fast_mode=fast_mode,
            )
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            trace = ExecutionTrace(
                task=task.value,
                selected_model=tool_spec.name,
                inputs_summary=f"{len(images)} images, config={input_config.value}",
                confidence=intent_conf,
                latency_ms=(time.perf_counter() - t_start) * 1000,
                status="error",
                error=str(e),
            )
            return AgentResponse(
                answer=f"Model execution error: {e}",
                confidence=0.0,
                trace=trace,
            )

        # 6. Build response
        latency = (time.perf_counter() - t_start) * 1000

        # Unpack result — tools return a dict with standardised keys
        answer = result.get("answer", "No answer generated.")
        confidence = result.get("confidence", intent_conf)
        visualisation = result.get("visualisation", None)
        change_mask = result.get("change_mask", None)
        class_map = result.get("class_map", None)
        bboxes = result.get("bboxes", None)

        trace = ExecutionTrace(
            task=task.value,
            selected_model=tool_spec.name,
            inputs_summary=(
                f"{len(images)} image(s) "
                f"[{', '.join(img.filename for img in images)}], "
                f"config={input_config.value}"
            ),
            confidence=confidence,
            latency_ms=round(latency, 2),
        )

        logger.info(
            f"Dispatch complete: model={tool_spec.name}, "
            f"confidence={confidence:.2f}, latency={latency:.1f}ms"
        )

        return AgentResponse(
            answer=answer,
            confidence=confidence,
            visualisation=visualisation,
            change_mask=change_mask,
            class_map=class_map,
            bboxes=bboxes,
            trace=trace,
        )

    # ── Batch Queries ─────────────────────────────────────────────────────

    def batch_dispatch(
        self,
        images: List[GeoImage],
        queries: List[str],
        fast_mode: bool = True,
    ) -> List[AgentResponse]:
        """Run multiple queries against the same image set."""
        return [self.dispatch(images, q, fast_mode=fast_mode) for q in queries]
