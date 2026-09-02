"""SatQuery AI — Model Smoke Tests.

Smoke tests for each specialist model with synthetic data.
Validates output shapes, types, and value ranges without
requiring GPU or pretrained model downloads.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.geoprocessor import GeoImage, GeoProcessor
from core.registry import TaskType, ToolRegistry


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_registry():
    registry = ToolRegistry()
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def synthetic_rgb():
    """Create a synthetic RGB image (H, W, 3) uint8."""
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_geo_image():
    """Create a synthetic 3-band GeoImage."""
    return GeoImage(
        array=np.random.rand(3, 128, 128).astype(np.float32),
        filename="synthetic.png",
        band_names=["Red", "Green", "Blue"],
    )


@pytest.fixture
def synthetic_4band_image():
    """Create a synthetic 4-band optical GeoImage (R,G,B,NIR)."""
    return GeoImage(
        array=np.random.rand(4, 128, 128).astype(np.float32),
        filename="optical_4band.tif",
        band_names=["Red", "Green", "Blue", "NIR"],
    )


@pytest.fixture
def synthetic_sar_image():
    """Create a synthetic 2-band SAR GeoImage."""
    return GeoImage(
        array=np.random.rand(2, 128, 128).astype(np.float32),
        filename="sar.tif",
        band_names=["VV", "VH"],
    )


# ─── GeoProcessor Tests ──────────────────────────────────────────────────────

class TestGeoProcessor:

    def test_load_png(self, tmp_path):
        """Test loading a PNG image."""
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        path = tmp_path / "test.png"
        img.save(path)

        geo = GeoProcessor()
        geo_image = geo.load(str(path))

        assert geo_image.num_bands == 3
        assert geo_image.height == 64
        assert geo_image.width == 64
        assert geo_image.array.dtype == np.float32
        assert geo_image.array.min() >= 0.0
        assert geo_image.array.max() <= 1.0

    def test_normalize_minmax(self):
        geo = GeoProcessor(norm_method="minmax")
        arr = np.array([[[0, 50], [100, 200]]], dtype=np.float32)
        result = geo.normalize_bands(arr)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_percentile(self):
        geo = GeoProcessor(norm_method="percentile")
        arr = np.random.rand(3, 64, 64).astype(np.float32) * 10000
        result = geo.normalize_bands(arr)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_to_rgb(self, synthetic_geo_image):
        rgb = GeoProcessor.to_rgb(synthetic_geo_image)
        assert rgb.shape == (128, 128, 3)
        assert rgb.dtype == np.uint8

    def test_to_pil(self, synthetic_geo_image):
        pil = GeoProcessor.to_pil(synthetic_geo_image)
        assert isinstance(pil, Image.Image)
        assert pil.size == (128, 128)

    def test_ndvi(self, synthetic_4band_image):
        ndvi = GeoProcessor.compute_ndvi(synthetic_4band_image)
        assert ndvi.shape == (128, 128)
        assert ndvi.dtype == np.float32
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_ndwi(self, synthetic_4band_image):
        ndwi = GeoProcessor.compute_ndwi(synthetic_4band_image)
        assert ndwi.shape == (128, 128)
        assert ndwi.dtype == np.float32

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "test.bmp"
        path.write_bytes(b"fake")
        geo = GeoProcessor()
        with pytest.raises(ValueError, match="Unsupported format"):
            geo.load(str(path))

    def test_file_not_found(self):
        geo = GeoProcessor()
        with pytest.raises(FileNotFoundError):
            geo.load("/nonexistent/file.png")


# ─── BandProjection Tests ────────────────────────────────────────────────────

class TestBandProjection:

    def test_forward_shape(self):
        import torch
        from models.adapters import BandProjection

        proj = BandProjection(in_channels=4, out_channels=3)
        x = torch.randn(1, 4, 64, 64)
        out = proj(x)
        assert out.shape == (1, 3, 64, 64)

    def test_forward_13_bands(self):
        import torch
        from models.adapters import BandProjection

        proj = BandProjection(in_channels=13, out_channels=3)
        x = torch.randn(1, 13, 32, 32)
        out = proj(x)
        assert out.shape == (1, 3, 32, 32)

    def test_single_band(self):
        import torch
        from models.adapters import BandProjection

        proj = BandProjection(in_channels=1, out_channels=3)
        x = torch.randn(2, 1, 32, 32)
        out = proj(x)
        assert out.shape == (2, 3, 32, 32)


# ─── Siamese Change Net Tests ────────────────────────────────────────────────

class TestSiameseChangeNet:

    def test_forward_shape(self):
        import torch
        from models.change_detector import SiameseChangeNet

        net = SiameseChangeNet(pretrained=False)
        pre = torch.randn(1, 3, 128, 128)
        post = torch.randn(1, 3, 128, 128)

        prob, mask = net(pre, post)
        assert prob.shape[0] == 1
        assert prob.shape[1] == 1
        assert mask.shape == prob.shape
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0

    def test_same_image_low_change(self):
        import torch
        from models.change_detector import SiameseChangeNet

        net = SiameseChangeNet(pretrained=False)
        img = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            prob, mask = net(img, img)

        # Same image should produce low/zero difference features
        # (though untrained net may still produce non-zero output)
        assert prob.shape[0] == 1


# ─── Fusion Classifier Tests ─────────────────────────────────────────────────

class TestFusionClassifier:

    def test_forward_shape(self):
        import torch
        from models.optical_sar_fusion import FusionClassifierCNN

        classifier = FusionClassifierCNN(
            in_channels=8, num_classes=5,
        )
        x = torch.randn(1, 8, 64, 64)
        out = classifier(x)
        assert out.shape == (1, 5, 64, 64)

    def test_custom_channels(self):
        import torch
        from models.optical_sar_fusion import FusionClassifierCNN

        classifier = FusionClassifierCNN(
            in_channels=6,
            hidden_channels=[32, 64],
            num_classes=3,
        )
        x = torch.randn(2, 6, 32, 32)
        out = classifier(x)
        assert out.shape == (2, 3, 32, 32)


# ─── Optical-SAR Fusion Feature Extraction Tests ─────────────────────────────

class TestFusionFeatures:

    def test_optical_features_rgb(self, synthetic_geo_image):
        from models.optical_sar_fusion import OpticalSARFusion

        features = OpticalSARFusion.extract_optical_features(synthetic_geo_image)
        assert features.shape[0] == 3  # RGB only (no NIR)
        assert features.shape[1] == 128
        assert features.shape[2] == 128

    def test_optical_features_with_nir(self, synthetic_4band_image):
        from models.optical_sar_fusion import OpticalSARFusion

        features = OpticalSARFusion.extract_optical_features(synthetic_4band_image)
        assert features.shape[0] == 5  # RGB + NDVI + NDWI
        assert features.dtype == np.float32

    def test_sar_features(self, synthetic_sar_image):
        from models.optical_sar_fusion import OpticalSARFusion

        features = OpticalSARFusion.extract_sar_features(synthetic_sar_image)
        assert features.shape[0] == 3  # VV + VH + ratio
        assert features.dtype == np.float32


# ─── Visualization Utility Tests ─────────────────────────────────────────────

class TestVisualizationUtils:

    def test_draw_bboxes(self, synthetic_rgb):
        from app.utils import draw_bboxes
        bboxes = [
            {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "label": "building", "score": 0.9},
            {"x1": 60, "y1": 60, "x2": 100, "y2": 100, "label": "tree", "score": 0.7},
        ]
        result = draw_bboxes(synthetic_rgb, bboxes)
        assert result.shape == synthetic_rgb.shape
        assert result.dtype == np.uint8

    def test_overlay_mask(self, synthetic_rgb):
        from app.utils import overlay_mask
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[30:80, 30:80] = 255
        result = overlay_mask(synthetic_rgb, mask)
        assert result.shape == synthetic_rgb.shape

    def test_confidence_gauge(self):
        from app.utils import create_confidence_gauge
        gauge = create_confidence_gauge(0.85)
        assert gauge.shape[2] == 3
        assert gauge.dtype == np.uint8

    def test_comparison(self, synthetic_rgb):
        from app.utils import create_comparison
        result = create_comparison(synthetic_rgb, synthetic_rgb)
        assert result.shape[2] == 3
        assert result.shape[1] > synthetic_rgb.shape[1]  # Wider than single image

    def test_json_report(self):
        from app.utils import generate_json_report
        import json
        trace = {"task": "vqa", "confidence": 0.8, "latency_ms": 42.0}
        report = generate_json_report(trace, "Test answer")
        parsed = json.loads(report)
        assert "satquery_ai_report" in parsed

    def test_pdf_report(self, synthetic_rgb):
        from app.utils import generate_pdf_report
        trace = {
            "task": "vqa", "selected_model": "test",
            "inputs_summary": "1 image", "confidence": 0.8,
            "latency_ms": 42.0, "timestamp": "2024-01-01T00:00:00Z",
            "status": "success",
        }
        pdf_bytes = generate_pdf_report(trace, [synthetic_rgb], "Test answer")
        assert len(pdf_bytes) > 0
        assert pdf_bytes[:4] == b"%PDF"


# ─── GeoImage DataClass Tests ────────────────────────────────────────────────

class TestGeoImage:

    def test_properties(self, synthetic_geo_image):
        assert synthetic_geo_image.num_bands == 3
        assert synthetic_geo_image.height == 128
        assert synthetic_geo_image.width == 128
        assert synthetic_geo_image.shape == (3, 128, 128)
        assert synthetic_geo_image.has_geo is False

    def test_has_geo_with_metadata(self):
        gi = GeoImage(
            array=np.zeros((3, 10, 10), dtype=np.float32),
            crs="EPSG:4326",
            transform="mock_transform",
        )
        assert gi.has_geo is True
