"""SatQuery AI — Visualization & Report Utilities.

Provides overlay rendering, PDF/JSON audit report generation,
and visualization helpers for the Streamlit app.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image


# ─── Bounding Box Drawing ────────────────────────────────────────────────────

def draw_bboxes(
    image: np.ndarray,
    bboxes: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 100),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw labeled bounding boxes on an image.

    Args:
        image: RGB image (H, W, 3) uint8.
        bboxes: List of dicts with keys: x1, y1, x2, y2, label, score.
        color: RGB color for boxes.
        thickness: Line thickness.
        font_scale: Font scale for labels.

    Returns:
        Image with drawn bboxes (H, W, 3) uint8.
    """
    vis = image.copy()
    for bbox in bboxes:
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        label = bbox.get("label", "")
        score = bbox.get("score", 0)

        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        label_text = f"{label} {score:.2f}" if score else label
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            vis, label_text,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    return vis


# ─── Mask Overlay ─────────────────────────────────────────────────────────────

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 50, 50),
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a semi-transparent mask on an image.

    Args:
        image: RGB image (H, W, 3) uint8.
        mask: Binary mask (H, W) uint8 {0, 255} or bool.
        color: RGB color for the overlay.
        alpha: Transparency (0 = transparent, 1 = opaque).

    Returns:
        Image with overlay (H, W, 3) uint8.
    """
    vis = image.copy()
    mask_bool = mask.astype(bool)

    overlay = np.zeros_like(vis)
    overlay[mask_bool] = color

    blended = cv2.addWeighted(vis, 1.0, overlay, alpha, 0)
    vis[mask_bool] = blended[mask_bool]

    return vis


# ─── Confidence Gauge ─────────────────────────────────────────────────────────

def create_confidence_gauge(
    confidence: float,
    width: int = 300,
    height: int = 40,
) -> np.ndarray:
    """Create a horizontal confidence gauge bar.

    Args:
        confidence: Value between 0 and 1.
        width: Image width.
        height: Image height.

    Returns:
        RGB image of the gauge (height, width, 3) uint8.
    """
    gauge = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark bg

    # Determine color: red → yellow → green
    if confidence < 0.3:
        color = (220, 50, 50)  # Red
    elif confidence < 0.6:
        color = (255, 180, 30)  # Yellow/Orange
    else:
        color = (50, 200, 80)  # Green

    # Draw filled bar
    bar_width = int(confidence * (width - 20))
    cv2.rectangle(gauge, (10, 8), (10 + bar_width, height - 8), color, -1)

    # Draw border
    cv2.rectangle(gauge, (10, 8), (width - 10, height - 8), (100, 100, 100), 1)

    # Draw text
    text = f"{confidence * 100:.0f}%"
    cv2.putText(
        gauge, text,
        (width // 2 - 20, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA,
    )

    return gauge


# ─── Side-by-Side Comparison ─────────────────────────────────────────────────

def create_comparison(
    img_left: np.ndarray,
    img_right: np.ndarray,
    label_left: str = "Before",
    label_right: str = "After",
    gap: int = 10,
) -> np.ndarray:
    """Create a side-by-side comparison image with labels.

    Args:
        img_left: Left image (H, W, 3) uint8.
        img_right: Right image (H, W, 3) uint8.
        label_left: Label for left image.
        label_right: Label for right image.
        gap: Gap between images in pixels.

    Returns:
        Combined image (H, W_total, 3) uint8.
    """
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    # Resize to same height
    target_h = max(h_l, h_r)
    if h_l != target_h:
        scale = target_h / h_l
        img_left = cv2.resize(img_left, (int(w_l * scale), target_h))
        w_l = img_left.shape[1]
    if h_r != target_h:
        scale = target_h / h_r
        img_right = cv2.resize(img_right, (int(w_r * scale), target_h))
        w_r = img_right.shape[1]

    # Create combined image with labels
    label_h = 30
    total_h = target_h + label_h
    total_w = w_l + gap + w_r
    combined = np.ones((total_h, total_w, 3), dtype=np.uint8) * 20

    # Add labels
    cv2.putText(
        combined, label_left,
        (w_l // 2 - 30, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (200, 200, 200), 1, cv2.LINE_AA,
    )
    cv2.putText(
        combined, label_right,
        (w_l + gap + w_r // 2 - 30, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (200, 200, 200), 1, cv2.LINE_AA,
    )

    # Paste images
    combined[label_h:label_h + target_h, :w_l] = img_left
    combined[label_h:label_h + target_h, w_l + gap:] = img_right

    return combined


# ─── PDF Report Generation ───────────────────────────────────────────────────

def clean_text_for_pdf(text: str) -> str:
    """Clean text so it's safe for FPDF standard core fonts."""
    if not text:
        return ""
    # Map common emojis / symbols to plain text
    replacements = {
        "📊": "[Stats]",
        "🛰️": "[SatQuery]",
        "🔍": "[Search]",
        "✅": "[OK]",
        "❌": "[Error]",
        "⚠️": "[Warning]",
        "🧠": "[AI]",
        "⚙️": "[Process]",
        "🎯": "[Result]",
        "📍": "[Location]",
        "🔄": "[Change]",
        "🌅": "[Optical]",
        "📡": "[SAR]",
        "—": "-",
        "–": "-",
        """: '"',
        """: '"',
        "'": "'",
        "'": "'",
        "…": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Encode to latin-1 with replacement for any remaining unrepresentable chars
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(
    trace: Dict[str, Any],
    images: List[np.ndarray],
    answer: str,
    output_path: Optional[str] = None,
) -> bytes:
    """Generate a PDF audit report for a query result.

    Args:
        trace: Execution trace dictionary.
        images: List of images used (RGB uint8).
        answer: The model's answer text.
        output_path: Optional file path to save the PDF.

    Returns:
        PDF bytes.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(37, 99, 235)  # Professional Sapphire Blue
    pdf.cell(0, 12, "SatQuery AI - Geospatial Intelligence Audit Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(3)

    # Metadata
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    timestamp = trace.get("timestamp", datetime.now(timezone.utc).isoformat())
    pdf.cell(0, 6, f"Generated: {clean_text_for_pdf(str(timestamp))}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # Horizontal line
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Execution Trace
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "Execution Trace", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 10)
    trace_fields = [
        ("Task", trace.get("task", "N/A")),
        ("Selected Model", trace.get("selected_model", "N/A")),
        ("Inputs", trace.get("inputs_summary", "N/A")),
        ("Confidence", f"{trace.get('confidence', 0) * 100:.1f}%"),
        ("Latency", f"{trace.get('latency_ms', 0):.1f} ms"),
        ("Status", trace.get("status", "N/A")),
    ]

    for label, value in trace_fields:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(45, 6, f"{label}:")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, clean_text_for_pdf(str(value)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(5)

    # Answer
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "Analysis Results", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 6, clean_text_for_pdf(answer))
    pdf.ln(5)

    # Images
    if images:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 8, "Input Imagery", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        for i, img in enumerate(images[:4]):
            pil_img = Image.fromarray(img)
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            max_w = 180
            aspect = img.shape[1] / img.shape[0]
            w = min(max_w, 90)
            h = w / aspect

            pdf.image(img_buffer, x=15, w=w, h=h)
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 5, f"Image {i+1}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)

    # Footer
    pdf.ln(8)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    pdf_bytes = bytes(pdf.output())

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"PDF report saved: {output_path}")

    return pdf_bytes


# ─── JSON Report Generation ──────────────────────────────────────────────────

def generate_json_report(
    trace: Dict[str, Any],
    answer: str,
    bboxes: Optional[List[Dict]] = None,
    class_distribution: Optional[Dict[str, float]] = None,
) -> str:
    """Generate a JSON audit report.

    Args:
        trace: Execution trace dictionary.
        answer: Model answer text.
        bboxes: Optional bounding boxes.
        class_distribution: Optional classification percentages.

    Returns:
        JSON string.
    """
    report = {
        "satquery_ai_report": {
            "version": "0.1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "execution_trace": trace,
            "result": {
                "answer": answer,
                "confidence": trace.get("confidence", 0),
            },
        }
    }

    if bboxes:
        report["satquery_ai_report"]["result"]["grounding"] = {
            "num_regions": len(bboxes),
            "bounding_boxes": bboxes,
        }

    if class_distribution:
        report["satquery_ai_report"]["result"]["classification"] = class_distribution

    return json.dumps(report, indent=2, default=str)


# ─── Image Format Conversion ─────────────────────────────────────────────────

def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy RGB array to PIL Image."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy RGB array."""
    return np.array(image.convert("RGB"))
