"""
SatQuery AI – FastAPI Backend
Wired to the real core.Orchestrator pipeline.
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# ── Core pipeline ─────────────────────────────────────────────────────────────

from core.geoprocessor import GeoProcessor
from core.orchestrator import Orchestrator

# Single shared instances (initialised once)
_geo = GeoProcessor()
_orchestrator = Orchestrator()


def _boot_models():
    """Instantiate all specialist models so they self-register into ToolRegistry."""
    try:
        from models.grounding import TextGuidedGrounder
        TextGuidedGrounder()
        logger.info("TextGuidedGrounder registered.")
    except Exception as e:
        logger.warning(f"TextGuidedGrounder load skipped: {e}")

    try:
        from models.vqa_caption import RSVQACaptioner
        RSVQACaptioner()
        logger.info("RSVQACaptioner registered.")
    except Exception as e:
        logger.warning(f"RSVQACaptioner load skipped: {e}")

    try:
        from models.change_detector import ChangeDetector
        ChangeDetector()
        logger.info("ChangeDetector registered.")
    except Exception as e:
        logger.warning(f"ChangeDetector load skipped: {e}")

    try:
        from models.optical_sar_fusion import OpticalSARFusion
        OpticalSARFusion()
        logger.info("OpticalSARFusion registered.")
    except Exception as e:
        logger.warning(f"OpticalSARFusion load skipped: {e}")

    logger.info(f"Registry summary:\n{_orchestrator.registry.summary()}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot all models at startup, clean up on shutdown."""
    logger.info("SatQuery AI backend starting — booting specialist models…")
    await asyncio.to_thread(_boot_models)
    logger.info("All models ready. API is live.")
    yield
    logger.info("SatQuery AI backend shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SatQuery AI Backend",
    description="Agentic Multimodal Remote Sensing Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", summary="Health check")
async def root() -> dict:
    return {"status": "online", "service": "SatQuery AI Backend"}


@app.post("/api/analyze", summary="Analyse satellite image")
async def analyze(
    file: UploadFile = File(..., description="Primary satellite image"),
    query: str = Form(..., description="Natural-language query"),
    file_b: Optional[UploadFile] = File(None, description="Optional secondary image for change-detection"),
) -> JSONResponse:
    """
    Accept an uploaded satellite image (and optional second image) plus a
    natural-language query, run the agentic RS pipeline via core.Orchestrator,
    and return structured analysis results.
    """
    t_start = time.perf_counter()
    tmp_paths: list[Path] = []

    try:
        # ── 1. Save uploaded file(s) to temp files ────────────────────────────
        suffix_a = Path(file.filename or "image.png").suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_a) as tmp_a:
            tmp_a.write(await file.read())
            tmp_paths.append(Path(tmp_a.name))

        if file_b and file_b.filename:
            suffix_b = Path(file_b.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_b) as tmp_b:
                tmp_b.write(await file_b.read())
                tmp_paths.append(Path(tmp_b.name))

        # ── 2. Load images via GeoProcessor ───────────────────────────────────
        images = []
        for i, p in enumerate(tmp_paths):
            try:
                geo_img = _geo.load(p)
                # Preserve original filename for orchestrator logging
                geo_img.filename = file.filename if i == 0 else (file_b.filename or p.name)
                images.append(geo_img)
            except Exception as e:
                logger.error(f"Failed to load image {p}: {e}")
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not load image '{p.name}': {e}"
                )

        # ── 3. Run the orchestrator ────────────────────────────────────────────
        logger.info(f"Running pipeline: query='{query}', images={[img.filename for img in images]}")
        agent_response = _orchestrator.dispatch(images=images, query=query, fast_mode=True)

        # ── 4. Build API response ──────────────────────────────────────────────
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        trace = agent_response.trace

        # Convert bboxes → frontend DetectionBox format [{label, confidence, box}]
        detections = []
        if agent_response.bboxes:
            for bbox in agent_response.bboxes:
                # grounding.py BBox.to_dict() returns x1/y1/x2/y2 + label + score
                if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                    box_coords = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                elif "box" in bbox:
                    box_coords = bbox["box"]
                else:
                    continue
                detections.append({
                    "label": bbox.get("label", "detection"),
                    "confidence": round(float(bbox.get("score", bbox.get("confidence", agent_response.confidence))), 4),
                    "box": [int(v) for v in box_coords],
                })

        response_body = {
            "status": "success" if not (trace and trace.status == "error") else "error",
            "task": trace.task if trace else "unknown",
            "query": query,
            "answer": agent_response.answer,
            "confidence": round(float(agent_response.confidence), 4),
            "execution_time_ms": latency_ms,
            "trace": {
                "selected_model": trace.selected_model if trace else "unknown",
                "modality": _orchestrator.detect_modality(images).value,
                "crs": images[0].crs or "N/A",
            },
            "detections": detections,
        }

        logger.info(
            f"Pipeline complete: task={response_body['task']}, "
            f"model={response_body['trace']['selected_model']}, "
            f"confidence={response_body['confidence']}, "
            f"latency={latency_ms}ms, detections={len(detections)}"
        )

        return JSONResponse(content=response_body)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    finally:
        # ── 5. Clean up temp files ─────────────────────────────────────────────
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
