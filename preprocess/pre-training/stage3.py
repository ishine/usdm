# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import json
import os
import random

import numpy as np
import psutil
from utils.multipack_sampler import MultipackDistributedBatchSampler
from tqdm import tqdm


def memory_usage_percentage():
    return psutil.virtual_memory().percent


def return_batch(sampler, data, output_path, max_length):
    total_batch = []
    part_number = 0

    for batch in tqdm(sampler):
        input_ids = []

        for idx in batch:
            input_ids += [int(i) for i in data[idx].split(" ")]

        assert len(input_ids) <= max_length
        input_ids = " ".join([str(i) for i in input_ids])

        total_batch.append(input_ids)

        if memory_usage_percentage() > 50:
            part_file_path = os.path.join(output_path, f"part{part_number}.txt")
            with open(part_file_path, "w") as part_file:
                part_file.write("\n".join(total_batch))
            total_batch = []
            part_number += 1

    if total_batch:
        final_part_file_path = os.path.join(output_path, f"part{part_number}.txt")
        with open(final_part_file_path, "w") as final_part_file:
            final_part_file.write("\n".join(total_batch))


def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input data directory.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output files.")
    parser.add_argument('--epoch', type=int, default=1,
                        help="Number of epochs for training or processing. Default is 1.")
    parser.add_argument('--max_length', type=int, default=8192,
                        help="Maximum sequence length for processing. Default is 8192.")
    parser.add_argument('--seed', type=int, default=43, help="Random seed for reproducibility. Default is 43.")

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_path, exist_ok=True)

    batches = []
    lengths = []
    data = []

    for epoch in range(1, args.epoch + 1):
        with open(os.path.join(args.data_path, f'data_epoch{epoch}.json'), "r") as length_json:
            lengths += json.load(length_json)

        data += [i.strip() for i in
                open(os.path.join(args.data_path, f'data_epoch{epoch}.txt'), "r").readlines()]

    paired_list = list(zip(data, lengths))
    paired_list = [(d, l) for (d, l) in paired_list if l <= args.max_length]
    random.shuffle(paired_list)
    unzipped_list = zip(*paired_list)
    data, lengths = [list(lst) for lst in unzipped_list]
    lengths = np.array(lengths)

    sampler = MultipackDistributedBatchSampler(lengths=lengths, batch_max_length=args.max_length, num_replicas=1, rank=0)
    sampler.set_epoch(0)
    del paired_list, unzipped_list, lengths

    return_batch(sampler, data, args.output_path, args.max_length)