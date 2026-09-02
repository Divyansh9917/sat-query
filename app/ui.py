"""SatQuery AI — Enterprise Geospatial Intelligence Application.

Executive Light-Theme UI for Earth Observation & Remote Sensing Analysis.
Designed for high legibility, crisp typography, dark high-contrast text,
and zero emoji clutter.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.geoprocessor import GeoImage, GeoProcessor
from core.orchestrator import Orchestrator, AgentResponse
from core.registry import ToolRegistry
from app.utils import (
    create_comparison,
    create_confidence_gauge,
    draw_bboxes,
    generate_json_report,
    generate_pdf_report,
    numpy_to_pil,
    overlay_mask,
)


# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="SatQuery AI | Earth Observation Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Professional Light Design System ─────────────────────────────────────────

def inject_custom_css():
    """Inject high-contrast executive light theme CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Reset & Global Typography ──────────────────── */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* ── Main Container ─────────────────────────────── */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1340px !important;
    }

    /* ── Explicit Dark Text for all standard headings ─ */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    p, span, label, li {
        color: #1e293b !important;
    }

    /* ── Top Executive Navbar ───────────────────────── */
    .executive-navbar {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0f172a;
        margin: 0;
        line-height: 1.1;
    }
    .brand-title span {
        color: #2563eb;
    }
    .brand-sub {
        font-size: 0.82rem;
        color: #475569;
        font-weight: 500;
        margin-top: 0.25rem;
        letter-spacing: 0.01em;
    }
    .nav-meta {
        display: flex;
        gap: 0.75rem;
        align-items: center;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #22c55e;
    }
    .mode-badge {
        display: inline-flex;
        align-items: center;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #1e293b;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
    }

    /* ── Section Cards ──────────────────────────────── */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .section-title {
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin-bottom: 0.85rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid #f1f5f9;
    }

    /* ── Analysis Result Container ──────────────────── */
    .analysis-container {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.25rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    .query-header-tag {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #475569;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .query-header-text {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .response-body-text {
        font-size: 0.98rem;
        line-height: 1.7;
        color: #1e293b;
        font-weight: 450;
    }

    /* ── Confidence Badges ──────────────────────────── */
    .conf-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .conf-high {
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .conf-med {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .conf-low {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }

    /* ── Trace Audit Card ───────────────────────────── */
    .audit-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #1e293b;
        line-height: 1.65;
    }
    .audit-k {
        width: 140px;
        display: inline-block;
        font-weight: 600;
        color: #475569;
    }
    .audit-v {
        font-weight: 600;
        color: #2563eb;
    }
    .audit-v-ok {
        font-weight: 700;
        color: #16a34a;
    }

    /* ── Sidebar Styles ─────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {
        color: #0f172a !important;
    }

    /* ── Primary Action Button ──────────────────────── */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.25) !important;
        transition: background-color 0.15s ease !important;
    }
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1e40af !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.35) !important;
    }

    /* ── Secondary Quick-Action Buttons ─────────────── */
    button[kind="secondary"], .stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        padding: 0.45rem 0.85rem !important;
        transition: all 0.15s ease !important;
    }
    button[kind="secondary"]:hover, .stButton > button[kind="secondary"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    /* ── Text Input Styling ─────────────────────────── */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        padding: 0.65rem 0.95rem !important;
    }
    .stTextInput input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }

    /* ── File Uploader ──────────────────────────────── */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px dashed #cbd5e1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] label, 
    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] div {
        color: #1e293b !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #2563eb !important;
    }

    /* ── Metrics ────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    /* ── Expander ───────────────────────────────────── */
    .streamlit-expanderHeader {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Helper Formatting Functions ──────────────────────────────────────────────

def confidence_pill_html(confidence: float) -> str:
    """Return high-contrast confidence pill HTML."""
    pct = confidence * 100
    if confidence >= 0.7:
        cls = "conf-high"
        label = "High Confidence"
    elif confidence >= 0.4:
        cls = "conf-med"
        label = "Medium Confidence"
    else:
        cls = "conf-low"
        label = "Low Confidence"
    return f'<span class="conf-pill {cls}">{pct:.0f}% {label}</span>'


def format_audit_trace(trace: dict) -> str:
    """Format execution trace as a clean, high-contrast monospace audit card."""
    error_line = ""
    if trace.get("error"):
        error_line = f'<div><span class="audit-k">Error:</span> <span style="color:#dc2626;font-weight:700;">{trace["error"]}</span></div>'

    status_cls = "audit-v-ok" if trace.get("status") == "success" else "audit-v"

    return f"""
    <div class="audit-box">
        <div><span class="audit-k">Task Type:</span> <span class="audit-v">{trace.get('task', 'N/A')}</span></div>
        <div><span class="audit-k">Active Model:</span> <span class="audit-v">{trace.get('selected_model', 'N/A')}</span></div>
        <div><span class="audit-k">Input Feeds:</span> <span>{trace.get('inputs_summary', 'N/A')}</span></div>
        <div><span class="audit-k">Confidence:</span> <span class="audit-v">{trace.get('confidence', 0) * 100:.1f}%</span></div>
        <div><span class="audit-k">Processing Time:</span> <span class="audit-v">{trace.get('latency_ms', 0):.2f} ms</span></div>
        <div><span class="audit-k">Timestamp (UTC):</span> <span>{trace.get('timestamp', 'N/A')}</span></div>
        <div><span class="audit-k">Status:</span> <span class="{status_cls}">{trace.get('status', 'N/A').upper()}</span></div>
        {error_line}
    </div>
    """


def load_uploaded_image(uploaded_file) -> Optional[GeoImage]:
    """Load uploaded file into a GeoImage instance."""
    if uploaded_file is None:
        return None

    try:
        geo = GeoProcessor()
        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix in (".tif", ".tiff"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            geo_image = geo.load(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
        else:
            img = Image.open(uploaded_file).convert("RGB")
            array = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            geo_image = GeoImage(
                array=array,
                filename=uploaded_file.name,
                band_names=["Red", "Green", "Blue"],
            )

        return geo_image

    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return None


# ─── Cached Initialization ────────────────────────────────────────────────────

@st.cache_resource
def init_orchestrator():
    """Initialize orchestrator and register specialist tools."""
    orchestrator = Orchestrator()

    try:
        from models.vqa_caption import RSVQACaptioner
        RSVQACaptioner()
    except Exception as e:
        st.warning(f"VQA model init skipped: {e}")

    try:
        from models.grounding import TextGuidedGrounder
        TextGuidedGrounder()
    except Exception as e:
        st.warning(f"Grounding model init skipped: {e}")

    try:
        from models.change_detector import ChangeDetector
        ChangeDetector()
    except Exception as e:
        st.warning(f"Change detector init skipped: {e}")

    try:
        from models.optical_sar_fusion import OpticalSARFusion
        OpticalSARFusion()
    except Exception as e:
        st.warning(f"Fusion model init skipped: {e}")

    return orchestrator


# ─── Sidebar View ─────────────────────────────────────────────────────────────

def render_sidebar() -> Tuple[str, list, bool]:
    """Render sidebar with mode selectors, uploaders, and system specs."""
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.4rem 0 1rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.2rem;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #0f172a; letter-spacing: -0.03em;">
                SATQUERY <span style="color: #2563eb;">AI</span>
            </div>
            <div style="font-size: 0.75rem; color: #475569; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.15rem;">
                Earth Observation Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Performance Mode Selector
        st.markdown("##### Execution Engine")
        engine_choice = st.radio(
            "Inference Mode:",
            [
                "Fast Analytics (Sub-Second)",
                "Deep Neural Network",
            ],
            index=0,
            help="Fast Analytics provides rapid spectral and analytical heuristics (<1s). Deep Neural uses deep language models.",
        )
        fast_mode = "Fast Analytics" in engine_choice

        st.divider()

        # Task Selection
        st.markdown("##### Analysis Mode")
        task_mode = st.radio(
            "Select Pipeline:",
            [
                "Single Image Analysis",
                "Bi-Temporal Pair (Change Detection)",
                "Cross-Modal Fusion (Optical + SAR)",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # Upload Imagery
        st.markdown("##### Imagery Upload")
        images = []

        if "Single Image" in task_mode:
            uploaded = st.file_uploader(
                "Upload Optical or SAR Image",
                type=["tif", "tiff", "png", "jpg", "jpeg"],
                key="single_upload",
                help="GeoTIFF, TIFF, PNG, JPEG formats supported",
            )
            if uploaded:
                images.append(uploaded)

        elif "Bi-Temporal" in task_mode:
            col1, col2 = st.columns(2)
            with col1:
                pre = st.file_uploader(
                    "Temporal T1 (Pre)",
                    type=["tif", "tiff", "png", "jpg", "jpeg"],
                    key="pre_upload",
                )
                if pre:
                    images.append(pre)
            with col2:
                post = st.file_uploader(
                    "Temporal T2 (Post)",
                    type=["tif", "tiff", "png", "jpg", "jpeg"],
                    key="post_upload",
                )
                if post:
                    images.append(post)

        elif "Cross-Modal" in task_mode:
            optical = st.file_uploader(
                "Optical Feed",
                type=["tif", "tiff", "png", "jpg", "jpeg"],
                key="optical_upload",
            )
            sar = st.file_uploader(
                "SAR Feed",
                type=["tif", "tiff", "png", "jpg", "jpeg"],
                key="sar_upload",
            )
            if optical:
                images.append(optical)
            if sar:
                images.append(sar)

        st.divider()

        # Tool Registry Spec
        st.markdown("##### Active Model Registry")
        registry = ToolRegistry()
        if len(registry) > 0:
            for tool in registry.list_tools():
                tasks_str = ", ".join(t.value for t in tool.tasks)
                st.markdown(
                    f'<div style="padding: 0.35rem 0; font-size: 0.78rem; border-bottom: 1px solid #f1f5f9;">'
                    f'<span style="color:#16a34a; font-weight:700;">●</span> '
                    f'<strong style="color:#0f172a;">{tool.name}</strong><br/>'
                    f'<span style="color: #475569; padding-left: 12px;">Tasks: {tasks_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<span style="color: #64748b; font-size: 0.8rem;">No models initialized</span>',
                unsafe_allow_html=True,
            )

        return task_mode, images, fast_mode


# ─── Main Panel View ──────────────────────────────────────────────────────────

def render_main_panel(
    task_mode: str,
    uploaded_files: list,
    fast_mode: bool,
    orchestrator: Orchestrator,
):
    """Render main panel with light theme cards and high-contrast dark text."""

    # Top Executive Navbar
    mode_text = "Fast Analytics Active (<1s)" if fast_mode else "Deep Neural Precision"
    st.markdown(f"""
    <div class="executive-navbar">
        <div>
            <h1 class="brand-title">SatQuery <span>AI</span></h1>
            <div class="brand-sub">
                Autonomous Vision-Language Intelligence System for Earth Observation & Remote Sensing
            </div>
        </div>
        <div class="nav-meta">
            <span class="mode-badge">{mode_text}</span>
            <div class="status-badge">
                <div class="status-pulse-dot"></div>
                System Active
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load uploaded images
    geo_images: List[GeoImage] = []
    for uf in uploaded_files:
        gi = load_uploaded_image(uf)
        if gi is not None:
            geo_images.append(gi)

    # ── Imagery Inspector ─────────────────────────────────────────────────
    if geo_images:
        st.markdown("##### Imagery Inspector")
        geo = GeoProcessor()

        if len(geo_images) == 1:
            rgb = geo.to_rgb(geo_images[0])
            st.image(rgb, caption=f"Active Input: {geo_images[0].filename}", use_container_width=True)

            with st.expander("Spectral & Coordinate Metadata", expanded=False):
                gi = geo_images[0]
                col1, col2, col3 = st.columns(3)
                col1.metric("Channels", gi.num_bands)
                col2.metric("Dimensions", f"{gi.width} x {gi.height} px")
                col3.metric("CRS Projection", gi.crs or "Local / Unprojected")

                if gi.band_names:
                    st.markdown(f"**Bands:** {', '.join(gi.band_names)}")

        elif len(geo_images) == 2:
            col1, col2 = st.columns(2)
            with col1:
                rgb1 = geo.to_rgb(geo_images[0])
                label1 = "Temporal T1 (Pre)" if "Bi-Temporal" in task_mode else "Optical Feed"
                st.image(rgb1, caption=f"{label1}: {geo_images[0].filename}", use_container_width=True)
            with col2:
                rgb2 = geo.to_rgb(geo_images[1])
                label2 = "Temporal T2 (Post)" if "Bi-Temporal" in task_mode else "SAR Feed"
                st.image(rgb2, caption=f"{label2}: {geo_images[1].filename}", use_container_width=True)

    # ── Geospatial Query Input ────────────────────────────────────────────
    st.markdown("##### Geospatial Query")

    if "Single Image" in task_mode:
        placeholder = "e.g., How many buildings are visible? / Describe this scene / Locate vegetation areas"
    elif "Bi-Temporal" in task_mode:
        placeholder = "e.g., What has changed? / Describe the key changes / Detect new structures"
    else:
        placeholder = "e.g., Identify water bodies and flood extent / Classify land cover distribution"

    query = st.text_input(
        "Enter your query:",
        placeholder=placeholder,
        label_visibility="collapsed",
        key="query_input",
    )

    # Preset Quick Actions
    quick_queries = {
        "Single": [
            "Describe this scene",
            "Identify built-up structures",
            "Locate vegetation canopy",
            "Analyze land cover profile",
        ],
        "Bi-Temporal": [
            "What has changed?",
            "Describe key changes",
            "Detect new construction",
            "Generate change map",
        ],
        "Cross-Modal": [
            "Detect water bodies",
            "Identify flood zones",
            "Classify land cover",
            "Analyze built-up areas",
        ],
    }

    mode_key = "Single" if "Single Image" in task_mode else "Bi-Temporal" if "Bi-Temporal" in task_mode else "Cross-Modal"
    quick_cols = st.columns(4)
    for i, q in enumerate(quick_queries.get(mode_key, [])):
        with quick_cols[i]:
            if st.button(q, key=f"quick_{i}", use_container_width=True, type="secondary"):
                query = q

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # ── Analysis Action Trigger ───────────────────────────────────────────
    if st.button("Run Analysis", use_container_width=True, type="primary"):
        if not geo_images:
            st.error("Please upload at least one image before executing analysis.")
            return

        if not query:
            st.error("Please enter a query or select a quick action above.")
            return

        progress_bar = st.progress(0, text="Initializing query classifier...")
        status_area = st.empty()

        try:
            t0 = time.perf_counter()

            # Intent classification
            progress_bar.progress(25, text="Classifying intent...")
            task, intent_conf = orchestrator.classify_intent(query)
            status_area.info(f"Target task: **{task.value}** (routing confidence: {intent_conf:.0%})")

            # Dispatch
            exec_label = "Running fast analytical pipeline..." if fast_mode else "Executing neural model inference..."
            progress_bar.progress(50, text=exec_label)

            response = orchestrator.dispatch(geo_images, query, fast_mode=fast_mode)

            elapsed = time.perf_counter() - t0
            progress_bar.progress(100, text=f"Analysis completed in {elapsed:.2f}s")

        except Exception as e:
            progress_bar.progress(100, text="Analysis failed")
            st.error(f"Analysis error encountered: {e}")
            return

        st.session_state["last_response"] = response
        st.session_state["last_query"] = query
        st.session_state["last_images"] = geo_images

    # ── Results Presentation ──────────────────────────────────────────────
    if "last_response" in st.session_state:
        response: AgentResponse = st.session_state["last_response"]
        display_query = st.session_state.get("last_query", "")

        st.markdown("---")
        st.markdown("##### Analysis Results")

        # Confidence Indicator
        st.markdown(confidence_pill_html(response.confidence), unsafe_allow_html=True)

        # High-Contrast Answer Card
        st.markdown(f"""
        <div class="analysis-container">
            <div class="query-header-tag">Evaluated Query</div>
            <div class="query-header-text">{display_query}</div>
            <div class="response-body-text">{response.answer}</div>
        </div>
        """, unsafe_allow_html=True)

        # Visualizations
        if response.visualisation is not None:
            st.markdown("###### Output Visualization")
            st.image(response.visualisation, use_container_width=True)

        # Bounding Boxes
        if response.bboxes:
            st.markdown("###### Detected Regions")
            for i, bbox in enumerate(response.bboxes):
                col1, col2 = st.columns([3, 1])
                col1.markdown(
                    f"**{bbox['label']}** - Coordinates: "
                    f"({bbox['x1']:.0f}, {bbox['y1']:.0f}) to "
                    f"({bbox['x2']:.0f}, {bbox['y2']:.0f})"
                )
                col2.markdown(f"Confidence Score: **{bbox['score']:.2f}**")

        # Change Mask Metric
        if response.change_mask is not None:
            st.markdown("###### Change Analysis")
            change_pct = np.mean(response.change_mask > 0) * 100
            st.metric("Total Changed Surface Area", f"{change_pct:.1f}%")

        # Audit Trace Card
        if response.trace:
            with st.expander("Audit Trail & Execution Trace", expanded=False):
                st.markdown(
                    format_audit_trace(response.trace.to_dict()),
                    unsafe_allow_html=True,
                )

        # ── Export Reports ────────────────────────────────────────────────
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("###### Certified Audit Export")
        export_cols = st.columns(2)

        with export_cols[0]:
            if response.trace:
                json_report = generate_json_report(
                    trace=response.trace.to_dict(),
                    answer=response.answer,
                    bboxes=response.bboxes,
                )
                st.download_button(
                    "Download JSON Audit Record",
                    data=json_report,
                    file_name="satquery_audit.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with export_cols[1]:
            if response.trace:
                geo = GeoProcessor()
                display_images = []
                for gi in st.session_state.get("last_images", []):
                    display_images.append(geo.to_rgb(gi))

                pdf_bytes = generate_pdf_report(
                    trace=response.trace.to_dict(),
                    images=display_images,
                    answer=response.answer,
                )
                st.download_button(
                    "Download PDF Executive Report",
                    data=bytes(pdf_bytes),
                    file_name="satquery_executive_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    inject_custom_css()

    orchestrator = init_orchestrator()

    task_mode, uploaded_files, fast_mode = render_sidebar()
    render_main_panel(task_mode, uploaded_files, fast_mode, orchestrator)


if __name__ == "__main__":
    main()
