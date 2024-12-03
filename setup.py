# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import sys
import subprocess
from pathlib import Path
from setuptools import find_namespace_packages, setup
from setuptools.command.install import install

def _init_submodules():
    subprocess.check_call(['git', 'submodule', 'init'])
    subprocess.check_call(['git', 'submodule', 'update'])

def _install_submodules():
    submodules = [
        'src/decoder',
    ]
    for submodule in submodules:
        submodule_path = Path(submodule)
        if not (submodule_path / 'setup.py').exists() and not (submodule_path / 'pyproject.toml').exists():
            raise FileNotFoundError(f"Neither 'setup.py' nor 'pyproject.toml' found in {submodule}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', submodule])

class CustomInstallCommand(install):
    def run(self):
        _install_submodules()
        super().run()

setup(
    name="USDM_trainer",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    python_requires=">=3.8",
    author="Heeseung Kim",
    url="https://github.com/naver-ai/usdm",
    install_requires=[
        'accelerate==1.0.1',
        'datasets==2.18.0',
        'deepspeed==0.15.2',
        'ninja==1.11.1.1',
        'packaging==23.2',
        'pandas==2.2.2',
        'peft==0.13.2',
        'psutil==6.1.0',
        'scikit-learn==1.5.2',
        'scipy==1.11.4',
        "seamless_communication @ git+https://github.com/facebookresearch/seamless_communication.git@90e2b57",
        'streamlit==1.39.0',
        'tensorboardx==2.6.2.2',
        'tokenizers==0.19.1',
        'torch==2.2.1',
        'transformers==4.40.2',
        'vllm==0.4.1'
    ],
    cmdclass={
        "install": CustomInstallCommand,
    },
)
