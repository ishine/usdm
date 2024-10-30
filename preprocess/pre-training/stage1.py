# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import glob
import librosa
import os
import string
import torch
from tqdm import tqdm

from seamless_communication.models.unit_extractor import UnitExtractor


def preprocess_text(text):
    text = text.lower()
    punctuation_to_remove = string.punctuation.replace("'", "")
    return text.translate(str.maketrans("", "", punctuation_to_remove)).strip(" ")


if __name__ == "__main__":
    # This code is based on LibriTTS dev-clean. Please modify it accordingly for other datasets.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to the directory containing the input wav files.")
    args = parser.parse_args()

    device = torch.device("cuda")

    unit_extractor = UnitExtractor("xlsr2_1b_v2", "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy", device=device)

    audio_paths = glob.glob(os.path.join(args.data_path, 'dev-clean/**/**/*.wav'))

    for audio_path in tqdm(audio_paths):
        audio, sr = librosa.load(audio_path, sr=16000)
        units = ' '.join([str(unit) for unit in unit_extractor.predict(torch.FloatTensor(audio).to(device), 35 - 1).cpu().tolist()])
        normalized_text = open(audio_path.replace('.wav', '.normalized.txt'), "r").readline()
        normalized_text = preprocess_text(normalized_text)

        with open(audio_path.replace('.wav', '.txt'), "w") as f:
            f.write(normalized_text)

        with open(audio_path.replace('.wav', '_unit.txt'), "w") as f:
            f.write(units)

