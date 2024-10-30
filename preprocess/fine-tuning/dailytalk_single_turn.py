# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import librosa
import os
import random
from seamless_communication.models.unit_extractor import UnitExtractor
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


MAX_LENGTH = 8192


def get_dict(metadata):
    dialog_dict = {}
    for fileline in metadata:
        dialog_id = fileline.split("|")[0]
        dialog_num = dialog_id.split("_")[-1]
        text = fileline.split("|")[-2]
        if dialog_num in dialog_dict:
            dialog_dict[dialog_num].append([dialog_id, text])
        else:
            dialog_dict[dialog_num] = [[dialog_id, text]]

    key_exception = []

    for key in dialog_dict:
        dialog_dict[key].sort(key=lambda x: int(x[0].split("_")[0]))
        assert len(dialog_dict[key]) - 1 == int(dialog_dict[key][-1][0].split("_")[0])
        spk_list = [i[0].split("_")[1] for i in dialog_dict[key]]

        if any(x == y for x, y in zip(spk_list, spk_list[1:])):
            key_exception.append(key)

    for key in key_exception:
        dialog_dict.pop(key)

    return dialog_dict


def return_speech_template(user_data, agent_data):
    _, user_unit, user_text = user_data
    _, agent_unit, agent_text = agent_data

    return (
        f"\n### User"
        f"\n{user_unit}<|correspond|>{user_text.lower()}"
        f"\n### Agent"
        f"\n{agent_text.lower()}<|correspond|>{agent_unit}"
        f"\n"
    )


def tokenize_speech(instruction_list, tokenizer):
    paired_data = []

    for instruction in instruction_list:
        input_ids = tokenizer(instruction).input_ids
        masks = input_ids[:]

        # newline index
        newline_indices = [index for index, value in enumerate(input_ids) if value == tokenizer("\n").input_ids[-1]]
        correspond_indices = [index for index, value in enumerate(input_ids) if
                              value == tokenizer("<|correspond|>").input_ids[-1]]

        # text, spoken response masking
        start_idx = 4
        while True:
            try:
                masks[newline_indices[start_idx] + 1:newline_indices[start_idx + 1]] = [- 100] * (
                            newline_indices[start_idx + 1] - newline_indices[start_idx] - 1)
                start_idx += 5
            except:
                break

        # speech recognition masking
        start_idx = 0
        while True:
            try:
                masks[correspond_indices[start_idx] + 1:newline_indices[round(2.5 * start_idx) + 3] + 1] = [- 100] * (
                            newline_indices[round(2.5 * start_idx) + 3] - correspond_indices[start_idx])
                start_idx += 2
            except:
                break

        labels = [elem if masks[i] == -100 else -100 for i, elem in enumerate(input_ids)]

        input_ids = ' '.join([str(i) for i in input_ids])
        labels = ' '.join([str(i) for i in labels])

        paired_data.append(input_ids + "|" + labels)
    return paired_data


def tokenize_metadata(metadata, tokenizer):
    data_list = []

    for user_data, agent_data in zip(metadata[0::2], metadata[1::2]):
        instruction = f"Below is a conversation between the user and the agent. Each turn includes the user's speech and its corresponding transcript, along with the agent's response text and the corresponding speech.\n"

        single_turn = return_speech_template(user_data, agent_data)

        if len(tokenizer(instruction + single_turn + '</s>').input_ids) <= MAX_LENGTH:
            instruction += single_turn + '</s>'

        if instruction != f"Below is a conversation between the user and the agent. Each turn includes the user's speech and its corresponding transcript, along with the agent's response text and the corresponding speech.\n":
            data_list.append(instruction)

    data_pair = tokenize_speech(data_list, tokenizer)

    return data_pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="naver-ai/USTM",
                        help="Name or path of the pre-trained tokenizer to load (e.g., 'naver-ai/USTM').")
    parser.add_argument('--model_cache_dir', type=str, required=True,
                        help="Directory to store tokenizer checkpoints and other cache files.")
    parser.add_argument('--train_metadata_path', type=str, default="../../dataset/fine-tuning/dailytalk/raw/train.txt",
                        help="Path to the metadata file for the training split of the dataset.")
    parser.add_argument('--test_metadata_path', type=str, default="../../dataset/fine-tuning/dailytalk/raw/test.txt",
                        help="Path to the metadata file for the test split of the dataset.")
    parser.add_argument('--output_path', type=str, default="../../dataset/fine-tuning/dailytalk/preprocessed",
                        help="Path to save the preprocessed dataset output files.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the directory containing the raw audio files.")
    args = parser.parse_args()

    train_metadata = [i.strip() for i in open(args.train_metadata_path, "r").readlines()]
    test_metadata = [i.strip() for i in open(args.test_metadata_path, "r").readlines()]

    train_dict = get_dict(train_metadata)
    test_dict = get_dict(test_metadata)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    os.makedirs(args.output_path, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    unit_extractor = UnitExtractor("xlsr2_1b_v2", "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy", device=device)

    train_dataset = []
    test_dataset = []

    for dialog_name in tqdm({**train_dict, **test_dict}):
        if dialog_name in train_dict:
            train = True
            metadata = train_dict[dialog_name]
        else:
            train = False
            metadata = test_dict[dialog_name]

        new_metadata = []

        for single_data in metadata:
            dialog_id, transcript = single_data
            wav_path = os.path.join(args.data_dir, f"{dialog_id.split('_')[-1][1:]}", f"{dialog_id}.wav")
            wav, sr = librosa.load(wav_path, sr=16000)
            units = ''.join(
                [f'<|unit{i}|>' for i in
                 unit_extractor.predict(torch.FloatTensor(wav).to(device), 35 - 1).cpu().tolist()]
            )
            new_metadata.append([dialog_id, units, transcript])

        even = True if len(new_metadata) % 2 == 0 else False

        even_metadata = tokenize_metadata(new_metadata, tokenizer)
        odd_metadata = tokenize_metadata(new_metadata[1:], tokenizer)

        if train:
            train_dataset += even_metadata
            train_dataset += odd_metadata
        else:
            test_dataset += even_metadata
            test_dataset += odd_metadata

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    with open(os.path.join(args.output_path, f"train.txt"), "w") as f:
        f.write("\n".join(train_dataset))

    with open(os.path.join(args.output_path, f"test.txt"), "w") as f:
        f.write("\n".join(test_dataset))




