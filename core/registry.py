"""SatQuery AI — Tool & Model Registry.

Singleton registry that specialist models self-register into.
The orchestrator queries this registry to match intents to tools.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger


# ─── Enums ────────────────────────────────────────────────────────────────────

class Modality(str, Enum):
    """Supported input modalities."""
    OPTICAL = "optical"
    SAR = "sar"
    BITEMPORAL = "bitemporal"
    CROSS_MODAL = "cross_modal"  # Optical + SAR together
    ANY = "any"


class TaskType(str, Enum):
    """Recognised task types the orchestrator can dispatch."""
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    CHANGE_CAPTION = "change_caption"
    CROSS_MODAL_FUSION = "cross_modal_fusion"


# ─── Tool Spec ────────────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    """Specification for a registered specialist tool.

    Attributes:
        name: Unique tool name (e.g. "rs_vqa_captioner").
        description: Human-readable description for the orchestrator.
        tasks: Set of TaskTypes this tool can handle.
        modalities: Set of Modalities this tool accepts.
        callable: The callable that executes the tool.
        priority: Higher = preferred when multiple tools match.
        version: Semver string for tracking.
        metadata: Arbitrary extra metadata (model size, etc.).
    """
    name: str
    description: str
    tasks: Set[TaskType]
    modalities: Set[Modality]
    callable: Callable[..., Any]
    priority: int = 0
    version: str = "0.1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def supports_task(self, task: TaskType) -> bool:
        return task in self.tasks

    def supports_modality(self, modality: Modality) -> bool:
        return Modality.ANY in self.modalities or modality in self.modalities


# ─── Tool Registry (Singleton) ───────────────────────────────────────────────

class ToolRegistry:
    """Thread-safe singleton registry for specialist tools.

    Usage::

        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="my_vqa",
            description="Visual Question Answering",
            tasks={TaskType.VQA, TaskType.CAPTION},
            modalities={Modality.OPTICAL, Modality.SAR},
            callable=my_vqa_function,
        ))

        # Look up tools
        tools = registry.match(TaskType.VQA, Modality.OPTICAL)
    """

    _instance: Optional[ToolRegistry] = None
    _lock = threading.Lock()

    def __new__(cls) -> ToolRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._tools: Dict[str, ToolSpec] = {}
        return cls._instance

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, spec: ToolSpec) -> None:
        """Register a tool spec. Overwrites if name already exists."""
        self._tools[spec.name] = spec
        logger.info(
            f"Registered tool: {spec.name} "
            f"(tasks={[t.value for t in spec.tasks]}, "
            f"modalities={[m.value for m in spec.modalities]})"
        )

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[ToolSpec]:
        """Get a tool by exact name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolSpec]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def match(
        self,
        task: TaskType,
        modality: Modality = Modality.ANY,
    ) -> List[ToolSpec]:
        """Find all tools matching a task and modality, sorted by priority (desc).

        Args:
            task: The task type to match.
            modality: The input modality to match.

        Returns:
            List of matching ToolSpecs, highest priority first.
        """
        matches = [
            spec
            for spec in self._tools.values()
            if spec.supports_task(task) and spec.supports_modality(modality)
        ]
        matches.sort(key=lambda s: s.priority, reverse=True)
        return matches

    def best_match(
        self,
        task: TaskType,
        modality: Modality = Modality.ANY,
    ) -> Optional[ToolSpec]:
        """Return the single best-matching tool, or None."""
        matches = self.match(task, modality)
        return matches[0] if matches else None

    # ── Utility ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all registered tools (useful for testing)."""
        self._tools.clear()
        logger.info("Registry cleared.")

    def summary(self) -> str:
        """Return a formatted summary of all registered tools."""
        if not self._tools:
            return "No tools registered."

        lines = ["┌─ Tool Registry ─────────────────────────────────"]
        for spec in self._tools.values():
            tasks_str = ", ".join(t.value for t in spec.tasks)
            mods_str = ", ".join(m.value for m in spec.modalities)
            lines.append(
                f"│ {spec.name} (v{spec.version}, priority={spec.priority})\n"
                f"│   Tasks: {tasks_str}\n"
                f"│   Modalities: {mods_str}\n"
                f"│   {spec.description}"
            )
        lines.append("└──────────────────────────────────────────────────")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools)"
