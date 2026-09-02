"""SatQuery AI — Geospatial Image Processor.

Handles loading, normalization, reprojection, and visualization of
GeoTIFF, TIFF, PNG, and JPEG remote-sensing imagery via rasterio.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image

# Rasterio is optional at import — gracefully degrade for PNG/JPEG-only usage
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn(
        "rasterio not installed — GeoTIFF support disabled. "
        "Install with: pip install rasterio",
        stacklevel=2,
    )


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class GeoImage:
    """Container for a loaded remote-sensing image with optional geo metadata."""

    array: np.ndarray  # Shape: (C, H, W) float32, values in [0, 1]
    filename: str = ""
    crs: Optional[str] = None  # e.g. "EPSG:4326"
    transform: Optional[object] = None  # rasterio Affine
    bounds: Optional[Tuple[float, float, float, float]] = None  # (left, bottom, right, top)
    band_names: list[str] = field(default_factory=list)
    nodata: Optional[float] = None
    original_dtype: str = "uint8"

    @property
    def num_bands(self) -> int:
        return self.array.shape[0]

    @property
    def height(self) -> int:
        return self.array.shape[1]

    @property
    def width(self) -> int:
        return self.array.shape[2]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.array.shape

    @property
    def has_geo(self) -> bool:
        return self.crs is not None and self.transform is not None


# ─── GeoProcessor ─────────────────────────────────────────────────────────────

NormMethod = Literal["minmax", "percentile", "histogram"]


class GeoProcessor:
    """Loads, normalizes, reprojects, and renders remote-sensing imagery."""

    SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        norm_method: NormMethod = "percentile",
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ):
        self.norm_method = norm_method
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

    # ── Loading ───────────────────────────────────────────────────────────

    def load(self, path: str | Path) -> GeoImage:
        """Load an image from disk. Auto-detects GeoTIFF vs raster image."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{suffix}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if suffix in (".tif", ".tiff") and HAS_RASTERIO:
            return self._load_geotiff(path)
        else:
            return self._load_standard(path)

    def _load_geotiff(self, path: Path) -> GeoImage:
        """Load a GeoTIFF with full geo-metadata preservation."""
        with rasterio.open(path) as src:
            array = src.read().astype(np.float32)  # (C, H, W)
            crs_str = str(src.crs) if src.crs else None
            transform = src.transform
            bounds = src.bounds
            band_names = [src.descriptions[i] or f"Band_{i+1}" for i in range(src.count)]
            nodata = src.nodata
            original_dtype = str(src.dtypes[0])

        logger.info(
            f"Loaded GeoTIFF: {path.name} | "
            f"shape={array.shape} | CRS={crs_str} | dtype={original_dtype}"
        )

        # Normalize to [0, 1]
        array = self.normalize_bands(array)

        return GeoImage(
            array=array,
            filename=path.name,
            crs=crs_str,
            transform=transform,
            bounds=tuple(bounds) if bounds else None,
            band_names=band_names,
            nodata=nodata,
            original_dtype=original_dtype,
        )

    def _load_standard(self, path: Path) -> GeoImage:
        """Load a standard PNG/JPEG image via PIL."""
        img = Image.open(path).convert("RGB")
        array = np.array(img, dtype=np.float32).transpose(2, 0, 1)  # (3, H, W)

        logger.info(f"Loaded image: {path.name} | shape={array.shape}")

        array = self.normalize_bands(array)

        return GeoImage(
            array=array,
            filename=path.name,
            band_names=["Red", "Green", "Blue"],
            original_dtype="uint8",
        )

    # ── Normalization ─────────────────────────────────────────────────────

    def normalize_bands(
        self,
        array: np.ndarray,
        method: NormMethod | None = None,
    ) -> np.ndarray:
        """Normalize band values to [0, 1] float32.

        Args:
            array: Input array of shape (C, H, W).
            method: Override the default normalization method.

        Returns:
            Normalized array of same shape, dtype float32, values in [0, 1].
        """
        method = method or self.norm_method
        result = np.empty_like(array, dtype=np.float32)

        for i in range(array.shape[0]):
            band = array[i].astype(np.float32)

            if method == "minmax":
                bmin, bmax = band.min(), band.max()
                if bmax - bmin > 0:
                    result[i] = (band - bmin) / (bmax - bmin)
                else:
                    result[i] = np.zeros_like(band)

            elif method == "percentile":
                p_lo = np.percentile(band, self.percentile_low)
                p_hi = np.percentile(band, self.percentile_high)
                if p_hi - p_lo > 0:
                    result[i] = np.clip((band - p_lo) / (p_hi - p_lo), 0, 1)
                else:
                    result[i] = np.zeros_like(band)

            elif method == "histogram":
                from skimage import exposure
                result[i] = exposure.equalize_hist(band)

            else:
                raise ValueError(f"Unknown normalization method: {method}")

        return result

    # ── Reprojection ──────────────────────────────────────────────────────

    def reproject_image(
        self,
        geo_image: GeoImage,
        target_crs: str = "EPSG:4326",
    ) -> GeoImage:
        """Reproject a GeoImage to the target CRS.

        Requires rasterio and a GeoImage with valid geo metadata.
        """
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio is required for reprojection.")
        if not geo_image.has_geo:
            logger.warning("Image has no geo metadata — skipping reprojection.")
            return geo_image

        src_crs = CRS.from_string(geo_image.crs)
        dst_crs = CRS.from_string(target_crs)

        if src_crs == dst_crs:
            logger.info("Image already in target CRS — no reprojection needed.")
            return geo_image

        transform, width, height = calculate_default_transform(
            src_crs,
            dst_crs,
            geo_image.width,
            geo_image.height,
            *geo_image.bounds,
        )

        dst_array = np.empty(
            (geo_image.num_bands, height, width), dtype=np.float32
        )

        for i in range(geo_image.num_bands):
            reproject(
                source=geo_image.array[i],
                destination=dst_array[i],
                src_transform=geo_image.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

        logger.info(
            f"Reprojected {geo_image.filename}: "
            f"{geo_image.crs} → {target_crs} | "
            f"new shape={dst_array.shape}"
        )

        return GeoImage(
            array=dst_array,
            filename=geo_image.filename,
            crs=target_crs,
            transform=transform,
            bounds=None,  # Would need to recompute
            band_names=geo_image.band_names,
            nodata=geo_image.nodata,
            original_dtype=geo_image.original_dtype,
        )

    # ── RGB Rendering ─────────────────────────────────────────────────────

    @staticmethod
    def to_rgb(
        geo_image: GeoImage,
        band_combo: Tuple[int, int, int] = (0, 1, 2),
    ) -> np.ndarray:
        """Render a 3-channel RGB uint8 image for display.

        Args:
            geo_image: The source GeoImage.
            band_combo: Tuple of (R, G, B) band indices (0-based).
                Common combos:
                  - True color:  (0, 1, 2) for RGB imagery
                  - False color: (3, 0, 1) for NIR-R-G (Sentinel-2)
                  - Agriculture: (3, 2, 1) for NIR-G-B

        Returns:
            uint8 numpy array of shape (H, W, 3).
        """
        num_bands = geo_image.num_bands
        r, g, b = band_combo

        # Clamp band indices
        r = min(r, num_bands - 1)
        g = min(g, num_bands - 1)
        b = min(b, num_bands - 1)

        rgb = np.stack(
            [geo_image.array[r], geo_image.array[g], geo_image.array[b]],
            axis=-1,
        )
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return rgb

    @staticmethod
    def to_pil(geo_image: GeoImage, band_combo: Tuple[int, int, int] = (0, 1, 2)) -> Image.Image:
        """Convert a GeoImage to a PIL Image for display."""
        rgb = GeoProcessor.to_rgb(geo_image, band_combo)
        return Image.fromarray(rgb)

    # ── Spectral Indices ──────────────────────────────────────────────────

    @staticmethod
    def compute_ndvi(geo_image: GeoImage, nir_band: int = 3, red_band: int = 0) -> np.ndarray:
        """Compute Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)
        """
        nir = geo_image.array[nir_band].astype(np.float64)
        red = geo_image.array[red_band].astype(np.float64)
        denominator = nir + red
        ndvi = np.where(denominator > 0, (nir - red) / denominator, 0.0)
        return ndvi.astype(np.float32)

    @staticmethod
    def compute_ndwi(geo_image: GeoImage, green_band: int = 1, nir_band: int = 3) -> np.ndarray:
        """Compute Normalized Difference Water Index.

        NDWI = (Green - NIR) / (Green + NIR)
        """
        green = geo_image.array[green_band].astype(np.float64)
        nir = geo_image.array[nir_band].astype(np.float64)
        denominator = green + nir
        ndwi = np.where(denominator > 0, (green - nir) / denominator, 0.0)
        return ndwi.astype(np.float32)
