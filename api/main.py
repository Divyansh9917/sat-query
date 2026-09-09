"""
SatQuery AI – FastAPI Backend
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SatQuery AI Backend",
    description="Agentic Multimodal Remote Sensing Analysis API",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────


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
    natural-language query, run the agentic RS pipeline, and return
    structured analysis results.

    Currently returns a deterministic mock response for front-end integration.
    Replace the body below with the real pipeline call once models are loaded.
    """
    # --- Real pipeline would go here ---
    # result = await pipeline.run(file, query, file_b)
    # -----------------------------------

    mock_response = {
        "status": "success",
        "task": "grounding",
        "query": query,
        "answer": "Identified candidate water bodies with high confidence.",
        "confidence": 0.86,
        "execution_time_ms": 94,
        "trace": {
            "selected_model": "GroundingDINO",
            "modality": "optical",
            "crs": "EPSG:4326",
        },
        "detections": [
            {
                "label": "water body",
                "confidence": 0.86,
                "box": [120, 80, 450, 320],
            }
        ],
    }

    return JSONResponse(content=mock_response)
