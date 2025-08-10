#!/usr/bin/env python3
"""
Setup script for ThetaIota - One Brain, Three Lobes
A native, on-device, self-reflective AI that learns, explains itself, and converses.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="thetaiota",
    version="1.0.0",
    author="ThetaIota Team",
    author_email="contact@thetaiota.ai",
    description="One Brain, Three Lobes - Native self-reflective AI with federation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Yufok1/thetaiota",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
        ],
        "optional": [
            "slowapi>=0.1.9",
            "starlette-exporter>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "thetaiota=cli_control:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai artificial-intelligence self-reflective transformer federation",
    project_urls={
        "Bug Reports": "https://github.com/Yufok1/thetaiota/issues",
        "Source": "https://github.com/Yufok1/thetaiota",
        "Documentation": "https://github.com/Yufok1/thetaiota#readme",
    },
)
