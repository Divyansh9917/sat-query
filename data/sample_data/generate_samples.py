#!/usr/bin/env python3
"""SatQuery AI — Generate Synthetic Sample Data.

Creates small synthetic GeoTIFF and PNG samples for testing
and demo purposes without requiring real satellite data.

Generates:
  - sample_optical.tif  — 4-band (R,G,B,NIR) synthetic optical scene
  - sample_sar.tif      — 2-band (VV, VH) synthetic SAR scene
  - sample_pre.png      — Pre-change RGB image
  - sample_post.png     — Post-change RGB image (with visible changes)

Usage:
    python data/sample_data/generate_samples.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_optical_tiff(path: Path, size: int = 256):
    """Generate a 4-band synthetic optical GeoTIFF (R, G, B, NIR)."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        # Create synthetic landscape
        h, w = size, size
        r = np.random.randint(40, 120, (h, w), dtype=np.uint16)
        g = np.random.randint(60, 180, (h, w), dtype=np.uint16)
        b = np.random.randint(30, 100, (h, w), dtype=np.uint16)
        nir = np.random.randint(100, 255, (h, w), dtype=np.uint16)

        # Add "vegetation" patches (high NIR, high green)
        for _ in range(8):
            cx, cy = np.random.randint(20, size - 20, 2)
            radius = np.random.randint(15, 40)
            yy, xx = np.ogrid[-radius:radius, -radius:radius]
            mask = xx**2 + yy**2 <= radius**2
            y0 = max(0, cy - radius)
            x0 = max(0, cx - radius)
            y1 = min(h, cy + radius)
            x1 = min(w, cx + radius)
            m = mask[:y1-y0, :x1-x0]
            g[y0:y1, x0:x1][m] = np.random.randint(140, 220)
            nir[y0:y1, x0:x1][m] = np.random.randint(180, 255)

        # Add "water" (low NIR, low values)
        water_y = slice(size // 2, size // 2 + 40)
        r[water_y, :] = np.random.randint(10, 40, (40, w), dtype=np.uint16)
        g[water_y, :] = np.random.randint(20, 60, (40, w), dtype=np.uint16)
        b[water_y, :] = np.random.randint(60, 120, (40, w), dtype=np.uint16)
        nir[water_y, :] = np.random.randint(5, 30, (40, w), dtype=np.uint16)

        # Add "urban" (bright, low NIR)
        urban_region = np.s_[10:60, 10:80]
        r[urban_region] = np.random.randint(150, 220, r[urban_region].shape, dtype=np.uint16)
        g[urban_region] = np.random.randint(140, 200, g[urban_region].shape, dtype=np.uint16)
        b[urban_region] = np.random.randint(130, 190, b[urban_region].shape, dtype=np.uint16)
        nir[urban_region] = np.random.randint(60, 100, nir[urban_region].shape, dtype=np.uint16)

        data = np.stack([r, g, b, nir])  # (4, H, W)

        transform = from_bounds(
            77.5, 12.9, 77.6, 13.0,  # Bangalore area coordinates
            w, h,
        )

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=4,
            dtype="uint16",
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(data)
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")
            dst.set_band_description(4, "NIR")

        print(f"  [+] Generated optical GeoTIFF: {path}")

    except ImportError:
        # Fallback: save as PNG
        r = np.random.randint(40, 200, (size, size), dtype=np.uint8)
        g = np.random.randint(60, 200, (size, size), dtype=np.uint8)
        b = np.random.randint(30, 180, (size, size), dtype=np.uint8)
        img = Image.fromarray(np.stack([r, g, b], axis=-1))
        png_path = path.with_suffix(".png")
        img.save(png_path)
        print(f"  [+] Generated optical PNG (rasterio not available): {png_path}")


def generate_sar_tiff(path: Path, size: int = 256):
    """Generate a 2-band synthetic SAR GeoTIFF (VV, VH)."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        h, w = size, size

        # VV backscatter (dB-like values)
        vv = np.random.uniform(-25, -5, (h, w)).astype(np.float32)
        vh = np.random.uniform(-30, -10, (h, w)).astype(np.float32)

        # Add "water" (very low backscatter)
        water_y = slice(size // 2, size // 2 + 40)
        vv[water_y, :] = np.random.uniform(-30, -25, (40, w)).astype(np.float32)
        vh[water_y, :] = np.random.uniform(-35, -28, (40, w)).astype(np.float32)

        # Add "urban" (high backscatter)
        urban = np.s_[10:60, 10:80]
        vv[urban] = np.random.uniform(-8, -2, vv[urban].shape).astype(np.float32)
        vh[urban] = np.random.uniform(-15, -8, vh[urban].shape).astype(np.float32)

        data = np.stack([vv, vh])

        transform = from_bounds(77.5, 12.9, 77.6, 13.0, w, h)

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=2,
            dtype="float32",
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(data)
            dst.set_band_description(1, "VV")
            dst.set_band_description(2, "VH")

        print(f"  [+] Generated SAR GeoTIFF: {path}")

    except ImportError:
        # Fallback: save synthetic SAR as PNG
        sar_img = Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
        sar_png = path.with_suffix(".png")
        sar_img.save(sar_png)
        print(f"  [+] Generated SAR PNG (rasterio not available): {sar_png}")


def generate_bitemporal_pair(dir_path: Path, size: int = 256):
    """Generate a bi-temporal PNG pair with visible changes."""
    # Pre-change: green landscape with some buildings
    pre = Image.new("RGB", (size, size), (60, 120, 50))
    draw_pre = ImageDraw.Draw(pre)

    # Add buildings
    for x, y, w_b, h_b in [(30, 30, 40, 35), (150, 80, 30, 25), (100, 180, 45, 30)]:
        draw_pre.rectangle([x, y, x + w_b, y + h_b], fill=(180, 170, 150))

    # Add road
    draw_pre.rectangle([0, size // 2 - 5, size, size // 2 + 5], fill=(100, 100, 100))

    # Add trees (circles)
    for _ in range(15):
        cx, cy = np.random.randint(10, size - 10, 2)
        r = np.random.randint(5, 15)
        draw_pre.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(30, 100 + np.random.randint(0, 50), 25))

    # Post-change: more buildings, some trees removed
    post = pre.copy()
    draw_post = ImageDraw.Draw(post)

    # New buildings (change)
    new_buildings = [(70, 120, 50, 40), (180, 30, 35, 45), (20, 180, 60, 35)]
    for x, y, w_b, h_b in new_buildings:
        draw_post.rectangle([x, y, x + w_b, y + h_b], fill=(200, 190, 170))

    # New road
    draw_post.rectangle([size // 2 - 5, 0, size // 2 + 5, size], fill=(110, 110, 110))

    # "Deforestation" — clear an area
    draw_post.rectangle([160, 160, 230, 230], fill=(150, 130, 90))

    pre.save(dir_path / "sample_pre.png")
    post.save(dir_path / "sample_post.png")
    print(f"  [+] Generated bi-temporal pair: sample_pre.png, sample_post.png")


def main():
    """Generate all sample data."""
    script_dir = Path(__file__).parent
    print("\n[*] Generating SatQuery AI sample data...\n")

    generate_optical_tiff(script_dir / "sample_optical.tif")
    generate_sar_tiff(script_dir / "sample_sar.tif")
    generate_bitemporal_pair(script_dir)

    print("\n[+] Sample data generation complete!")
    print(f"    Location: {script_dir.resolve()}\n")


if __name__ == "__main__":
    main()
