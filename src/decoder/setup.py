# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

from setuptools import setup

setup(
    name="voicebox",
    py_modules=["voicebox"],
    install_requires=[
        "librosa",
        "matplotlib",
        "tensorboard",
        "torch",
        "torchvision",
        "torchaudio",
        "torchcde",
        "tqdm"
    ],
)
