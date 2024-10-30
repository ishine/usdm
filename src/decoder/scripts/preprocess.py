# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import glob
import librosa
import os
import random
from seamless_communication.models.unit_extractor import UnitExtractor
import torch
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True, help="Name of the dataset to be processed.")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to the directory containing the dataset files.")
    parser.add_argument('--data_split', type=str, required=True,
                        help="Dataset split (e.g., 'train', 'valid', 'test').")
    args = parser.parse_args()

    audio_list = glob.glob(os.path.join(args.data_path, "**/*.wav"), recursive=True) + glob.glob(os.path.join(args.data_path, "**/*.flac"), recursive=True) + glob.glob(os.path.join(args.data_path, "**/*.mp3"), recursive=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    unit_extractor = UnitExtractor("xlsr2_1b_v2",
                                   "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
                                   device=device)

    filelists = []

    for audio_path in tqdm(audio_list):
        wav, sr = librosa.load(audio_path, sr=16000)
        units = [str(i) for i in unit_extractor.predict(torch.FloatTensor(wav).to(device), 35 - 1).cpu().tolist()]
        durations = [str(1) for _ in units]
        filelists.append(audio_path + "|" + " ".join(units) + "|" + " ".join(durations))

    random.shuffle(filelists)

    output_path = f"voicebox/filelists/{args.data_name}/{args.data_split}.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(filelists))
