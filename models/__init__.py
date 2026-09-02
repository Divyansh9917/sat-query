"""SatQuery AI — Specialist Models Package.

Each model self-registers into the ToolRegistry on first instantiation.
Imports are lazy to avoid loading heavyweight model weights on package import.
"""

__all__ = [
    "RSLoRAAdapter",
    "BandProjection",
    "RSVQACaptioner",
    "TextGuidedGrounder",
    "ChangeDetector",
    "OpticalSARFusion",
]


def __getattr__(name: str):
    """Lazy-load model classes to avoid pulling in torch at import time."""
    if name in ("RSLoRAAdapter", "BandProjection"):
        from models.adapters import RSLoRAAdapter, BandProjection
        return {"RSLoRAAdapter": RSLoRAAdapter, "BandProjection": BandProjection}[name]
    if name == "RSVQACaptioner":
        from models.vqa_caption import RSVQACaptioner
        return RSVQACaptioner
    if name == "TextGuidedGrounder":
        from models.grounding import TextGuidedGrounder
        return TextGuidedGrounder
    if name == "ChangeDetector":
        from models.change_detector import ChangeDetector
        return ChangeDetector
    if name == "OpticalSARFusion":
        from models.optical_sar_fusion import OpticalSARFusion
        return OpticalSARFusion
    raise AttributeError(f"module 'models' has no attribute {name!r}")
