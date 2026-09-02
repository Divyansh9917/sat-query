"""SatQuery AI — Agentic Multimodal Remote-Sensing Assistant."""

from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="satquery-ai",
    version="0.1.0",
    author="SatQuery AI Team",
    description="Agentic multimodal remote-sensing assistant for VQA, "
                "grounding, change detection, and optical-SAR fusion.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/satquery-ai/satquery-ai",
    packages=find_packages(exclude=["tests*", "data*"]),
    python_requires=">=3.10",
    install_requires=[
        "rasterio>=1.3.9",
        "pyproj>=3.6.0",
        "shapely>=2.0.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0",
        "peft>=0.11.0",
        "accelerate>=0.30.0",
        "open-clip-torch>=2.24.0",
        "opencv-python>=4.9.0",
        "Pillow>=10.2.0",
        "scikit-image>=0.22.0",
        "streamlit>=1.35.0",
        "plotly>=5.20.0",
        "fpdf2>=2.7.0",
        "jinja2>=3.1.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "satquery=app.ui:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
    ],
)
