from setuptools import setup, find_packages

setup(
    name='experitur',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'regex',
        'timer_cm',
        'etaprogress',
        'pyyaml',
        'sklearn',
    ],
    entry_points={
        'console_scripts': ['experitur=experitur.cli:cli'],
    },
)
