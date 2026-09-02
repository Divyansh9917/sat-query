"""SatQuery AI — Agent & Orchestrator Tests.

Tests intent classification, input validation, dispatch routing,
modality detection, and execution trace generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.geoprocessor import GeoImage
from core.orchestrator import (
    AgentResponse,
    ExecutionTrace,
    InputConfig,
    Orchestrator,
)
from core.registry import Modality, TaskType, ToolRegistry, ToolSpec


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the registry before each test."""
    registry = ToolRegistry()
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def orchestrator():
    return Orchestrator()


@pytest.fixture
def dummy_optical_image():
    """Create a dummy 3-band optical GeoImage."""
    return GeoImage(
        array=np.random.rand(3, 64, 64).astype(np.float32),
        filename="test_optical.png",
        band_names=["Red", "Green", "Blue"],
    )


@pytest.fixture
def dummy_sar_image():
    """Create a dummy 2-band SAR GeoImage."""
    return GeoImage(
        array=np.random.rand(2, 64, 64).astype(np.float32),
        filename="test_sar.tif",
        band_names=["VV", "VH"],
    )


@pytest.fixture
def dummy_tool():
    """Create a dummy tool for registry tests."""
    def mock_callable(images, query, task, **kwargs):
        return {
            "answer": f"Mock answer for: {query}",
            "confidence": 0.85,
        }

    return ToolSpec(
        name="mock_vqa",
        description="Mock VQA tool for testing",
        tasks={TaskType.VQA, TaskType.CAPTION},
        modalities={Modality.OPTICAL, Modality.ANY},
        callable=mock_callable,
        priority=5,
    )


# ─── Intent Classification Tests ─────────────────────────────────────────────

class TestIntentClassification:

    def test_vqa_question(self, orchestrator):
        task, conf = orchestrator.classify_intent("How many buildings are in this image?")
        assert task == TaskType.VQA
        assert conf > 0.3

    def test_caption_query(self, orchestrator):
        task, conf = orchestrator.classify_intent("Describe this satellite scene.")
        assert task == TaskType.CAPTION
        assert conf > 0.3

    def test_grounding_query(self, orchestrator):
        task, conf = orchestrator.classify_intent("Where are the trees located?")
        assert task == TaskType.GROUNDING
        assert conf > 0.3

    def test_change_detection_query(self, orchestrator):
        task, conf = orchestrator.classify_intent("Detect changes between the two images.")
        assert task == TaskType.CHANGE_DETECTION
        assert conf > 0.3

    def test_change_caption_query(self, orchestrator):
        task, conf = orchestrator.classify_intent("Describe what has changed in the scene.")
        assert task == TaskType.CHANGE_CAPTION
        assert conf > 0.3

    def test_cross_modal_query(self, orchestrator):
        task, conf = orchestrator.classify_intent("Identify flood zones using SAR data.")
        assert task == TaskType.CROSS_MODAL_FUSION
        assert conf > 0.3

    def test_ambiguous_defaults_to_vqa(self, orchestrator):
        task, conf = orchestrator.classify_intent("Hello world")
        assert task == TaskType.VQA
        assert conf == 0.5  # Default confidence for fallback

    def test_question_mark_triggers_vqa(self, orchestrator):
        task, _ = orchestrator.classify_intent("Is this urban?")
        assert task == TaskType.VQA


# ─── Modality Detection Tests ────────────────────────────────────────────────

class TestModalityDetection:

    def test_single_optical(self, dummy_optical_image):
        mod = Orchestrator.detect_modality([dummy_optical_image])
        assert mod == Modality.OPTICAL

    def test_single_sar(self, dummy_sar_image):
        mod = Orchestrator.detect_modality([dummy_sar_image])
        assert mod == Modality.SAR

    def test_bitemporal(self, dummy_optical_image):
        mod = Orchestrator.detect_modality([dummy_optical_image, dummy_optical_image])
        assert mod == Modality.BITEMPORAL

    def test_cross_modal(self, dummy_optical_image, dummy_sar_image):
        mod = Orchestrator.detect_modality([dummy_optical_image, dummy_sar_image])
        assert mod == Modality.CROSS_MODAL

    def test_empty_images(self):
        mod = Orchestrator.detect_modality([])
        assert mod == Modality.ANY


# ─── Input Validation Tests ──────────────────────────────────────────────────

class TestInputValidation:

    def test_single_image_valid(self, dummy_optical_image):
        config = Orchestrator.validate_inputs(TaskType.VQA, [dummy_optical_image])
        assert config == InputConfig.SINGLE_OPTICAL

    def test_single_sar_detected(self, dummy_sar_image):
        config = Orchestrator.validate_inputs(TaskType.VQA, [dummy_sar_image])
        assert config == InputConfig.SINGLE_SAR

    def test_vqa_no_images_raises(self):
        with pytest.raises(ValueError, match="requires at least 1"):
            Orchestrator.validate_inputs(TaskType.VQA, [])

    def test_change_needs_two(self, dummy_optical_image):
        with pytest.raises(ValueError, match="requires 2"):
            Orchestrator.validate_inputs(TaskType.CHANGE_DETECTION, [dummy_optical_image])

    def test_bitemporal_valid(self, dummy_optical_image):
        config = Orchestrator.validate_inputs(
            TaskType.CHANGE_DETECTION,
            [dummy_optical_image, dummy_optical_image],
        )
        assert config == InputConfig.BITEMPORAL_PAIR

    def test_fusion_needs_two(self, dummy_optical_image):
        with pytest.raises(ValueError, match="requires both"):
            Orchestrator.validate_inputs(TaskType.CROSS_MODAL_FUSION, [dummy_optical_image])


# ─── Registry Tests ──────────────────────────────────────────────────────────

class TestToolRegistry:

    def test_register_and_get(self, dummy_tool):
        registry = ToolRegistry()
        registry.register(dummy_tool)
        assert "mock_vqa" in registry
        assert registry.get("mock_vqa") == dummy_tool

    def test_match_by_task(self, dummy_tool):
        registry = ToolRegistry()
        registry.register(dummy_tool)
        matches = registry.match(TaskType.VQA)
        assert len(matches) == 1
        assert matches[0].name == "mock_vqa"

    def test_no_match(self, dummy_tool):
        registry = ToolRegistry()
        registry.register(dummy_tool)
        matches = registry.match(TaskType.CHANGE_DETECTION)
        assert len(matches) == 0

    def test_best_match(self, dummy_tool):
        registry = ToolRegistry()
        registry.register(dummy_tool)
        best = registry.best_match(TaskType.VQA, Modality.OPTICAL)
        assert best is not None
        assert best.name == "mock_vqa"

    def test_priority_ordering(self):
        registry = ToolRegistry()

        def noop(**kw): return {}

        low = ToolSpec("low", "Low priority", {TaskType.VQA}, {Modality.ANY}, noop, priority=1)
        high = ToolSpec("high", "High priority", {TaskType.VQA}, {Modality.ANY}, noop, priority=10)

        registry.register(low)
        registry.register(high)
        best = registry.best_match(TaskType.VQA)
        assert best.name == "high"

    def test_unregister(self, dummy_tool):
        registry = ToolRegistry()
        registry.register(dummy_tool)
        registry.unregister("mock_vqa")
        assert "mock_vqa" not in registry

    def test_summary(self, dummy_tool):
        registry = ToolRegistry()
        registry.register(dummy_tool)
        s = registry.summary()
        assert "mock_vqa" in s


# ─── Dispatch Tests ──────────────────────────────────────────────────────────

class TestDispatch:

    def test_dispatch_with_registered_tool(self, orchestrator, dummy_optical_image, dummy_tool):
        registry = ToolRegistry()
        registry.register(dummy_tool)

        response = orchestrator.dispatch(
            [dummy_optical_image],
            "How many buildings are visible?",
        )

        assert isinstance(response, AgentResponse)
        assert response.answer.startswith("Mock answer")
        assert response.confidence == 0.85
        assert response.trace is not None
        assert response.trace.status == "success"
        assert response.trace.latency_ms > 0

    def test_dispatch_no_tool_returns_error(self, orchestrator, dummy_optical_image):
        response = orchestrator.dispatch(
            [dummy_optical_image],
            "How many buildings?",
        )
        assert "no specialist model" in response.answer.lower() or response.confidence == 0.0

    def test_dispatch_invalid_inputs(self, orchestrator):
        response = orchestrator.dispatch(
            [],
            "How many buildings?",
        )
        assert response.confidence == 0.0
        assert response.trace.status == "error"

    def test_dispatch_trace_fields(self, orchestrator, dummy_optical_image, dummy_tool):
        registry = ToolRegistry()
        registry.register(dummy_tool)

        response = orchestrator.dispatch(
            [dummy_optical_image],
            "Describe this scene",
        )

        trace = response.trace
        assert trace.task in [t.value for t in TaskType]
        assert trace.selected_model == "mock_vqa"
        assert trace.timestamp  # Not empty
        assert trace.latency_ms >= 0


# ─── Execution Trace Tests ────────────────────────────────────────────────────

class TestExecutionTrace:

    def test_trace_to_dict(self):
        trace = ExecutionTrace(
            task="vqa",
            selected_model="test_model",
            inputs_summary="1 image",
            confidence=0.9,
            latency_ms=42.5,
        )
        d = trace.to_dict()
        assert d["task"] == "vqa"
        assert d["confidence"] == 0.9
        assert d["latency_ms"] == 42.5

    def test_agent_response_to_dict(self):
        trace = ExecutionTrace(
            task="caption", selected_model="blip2",
            inputs_summary="1 image", confidence=0.85,
            latency_ms=100.0,
        )
        resp = AgentResponse(answer="A city scene.", confidence=0.85, trace=trace)
        d = resp.to_dict()
        assert d["answer"] == "A city scene."
        assert "trace" in d
