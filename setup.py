from setuptools import setup, find_packages

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name='experitur',
    version='1.0.0a1',
    author='Simon-Martin Schroeder',
    author_email="martin.schroeder@nerdluecht.de",
    description="Automates machine learning and other computer experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moi90/experitur",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'regex',
        'tqdm',
        'pyyaml',
        'pandas',
    ],
    python_requires='>=3.5',
    entry_points={
        'console_scripts': ['experitur=experitur.cli:cli'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
    ],
)
