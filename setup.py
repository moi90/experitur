from setuptools import find_packages, setup

import versioneer

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name="experitur",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Simon-Martin Schroeder",
    author_email="martin.schroeder@nerdluecht.de",
    description="Automates machine learning and other computer experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moi90/experitur",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Click",
        "regex",
        "tqdm",
        "pyyaml",
        "unavailable-object",
        "filelock",
        "cachetools",
        "joblib",  # joblib.dump/load for compressed checkpoints
        "termcolor",  # for colorful terminal output
        "dictdiffer",
        "numpy>=1.20",
        "pandas>=1.3",
    ],
    python_requires=">=3.8",
    extras_require={
        "tests": ["pytest", "pytest-cov", "pytest-benchmark"],
        "optional": ["scikit-optimize", "scikit-learn", "matplotlib", "natsort"],
        "docs": [
            "sphinx >= 1.4",
            "sphinx_rtd_theme",
            "sphinxcontrib-programoutput",
            "sphinx-autodoc-typehints>=1.10.0",
        ],
        "dev": ["black", "mypy", "pydocstyle", "pylint", "flake8"],
    },
    entry_points={"console_scripts": ["experitur=experitur.cli:cli"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
    ],
)
