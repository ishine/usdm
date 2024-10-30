# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import random
import datasets
import numpy as np
import torch


def mapping_fn(example):
    input_ids = example["text"]
    input_ids = [int(i) for i in input_ids.split(" ")]
    example["input_ids"] = input_ids
    del example["text"]
    return example


def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def main(args: argparse.Namespace):
    # Here we are passing lists of train and validation files directly
    raw_datasets = datasets.load_dataset(
        path="text",
        data_files={
            "train": args.train_dataset_path,
            "test": args.validation_dataset_path
        },
        cache_dir=args.cache_path,
        num_proc=16
    )

    raw_datasets = raw_datasets.map(mapping_fn, num_proc=16)
    raw_datasets.save_to_disk(dataset_dict_path=f"{args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, nargs='+', required=True,
                        help="List of paths for training datasets. Multiple files can be specified by separating them with spaces.")
    parser.add_argument("--validation_dataset_path", type=str, nargs='+', required=True,
                        help="List of paths for validation datasets. Multiple files can be specified by separating them with spaces.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output files.")
    parser.add_argument("--cache_path", type=str, required=True, help="Path to store cached data for faster access.")
    parser.add_argument('--seed', type=int, default=43, help="Random seed for reproducibility. Default is 43.")

    args, _ = parser.parse_known_args()
    set_seed(args.seed)
    main(args)
