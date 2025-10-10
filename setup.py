#!/usr/bin/env python3
"""
Setup script for ds-covid package
Alternative to pyproject.toml for broader compatibility
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ds-covid",
    version="0.1.0",
    author="Rafael Cepa, Cirine Moire, Steven Moire",
    author_email="rafael.cepa@cnrs-orleans.fr",
    description="COVID-19 Radiography Analysis Package for Data Science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/L-Poca/DS_COVID",
    project_urls={
        "Bug Tracker": "https://github.com/L-Poca/DS_COVID/issues",
        "Documentation": "https://github.com/L-Poca/DS_COVID/blob/main/README.md",
        "Source Code": "https://github.com/L-Poca/DS_COVID",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "ds_covid": ["*.yml", "*.yaml", "*.json", "config/*"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "pillow>=8.3.0",
        "tqdm>=4.62.0",
        "plotly>=5.0.0",
        "streamlit>=1.10.0",
        "scikit-image>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "ruff>=0.1.0",
            "mypy>=0.991",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "colab": [
            "google-colab",
            "kaggle>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ds-covid-train=ds_covid.cli:train_model",
            "ds-covid-predict=ds_covid.cli:predict",
            "ds-covid-apply-masks=ds_covid.cli:apply_masks",
            "ds-covid-streamlit=ds_covid.cli:run_streamlit",
        ],
    },
    keywords="covid-19 radiography machine-learning medical-imaging deep-learning",
    zip_safe=False,
)