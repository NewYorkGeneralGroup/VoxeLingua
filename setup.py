from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="voxelingua",
    version="0.1.0",
    author="Advanced AI Research Team",
    author_email="research@voxelingua.ai",
    description="Advanced voxel-based language processing system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/voxelingua/voxelingua",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.2",
        "transformers>=4.5.0",
        "wandb>=0.12.0",
        "h5py>=3.1.0",
        "tqdm>=4.62.0",
        "pytest>=6.2.4",
        "mypy>=0.910",
        "black>=21.5b2",
        "isort>=5.9.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
            "flake8",
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
)
