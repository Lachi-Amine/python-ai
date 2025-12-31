"""
Setup Script for Blind Path Detection System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="blind-path-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive blind path obstacle detection system with audio navigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/blind-path-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy>=1.21.0",
        "opencv-python>=4.6.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "pyttsx3>=2.90",
        "pyaudio>=0.2.11",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "audio": [
            "sounddevice>=0.4.4",
            "soundfile>=0.10.3",
            "pygame>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "blind-path-train=blind_path_detection.train:main",
            "blind-path-eval=blind_path_detection.evaluate:main",
            "blind-path-demo=blind_path_detection.main:run_webcam_demo",
        ],
    },
)