# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import glob
import json
from multiprocessing.pool import Pool
import os
import random
import scipy.stats as stats
import textgrid
from tokenizers import AddedToken
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer


NUM_UNIT_TOKENS = 10000
SEGMENT_SECOND = 10


def align_text_to_sequence(textgrid_path, epoch):
    metadata_path = textgrid_path.replace('.TextGrid', '_unit.txt').replace('-textgrid', '')

    if not os.path.exists(metadata_path):
        return None, None

    metadata = open(metadata_path, "r").readline()
    sequence = tokenizer(''.join([f"<|unit{i}|>" for i in metadata.split("\t")[0].split(" ")])).input_ids[1:]

    # Load TextGrid file
    tg = textgrid.TextGrid.fromFile(textgrid_path)

    # Call the word alignment part
    word_alignments = tg.getFirst('words')  # 'words' is the name of the word tier in the TextGrid

    # Handle blank spaces
    intervals = word_alignments.intervals
    ratio = len(sequence) / intervals[-1].maxTime

    transcript = " ".join([i.mark for i in intervals if i.mark != ''])

    for i in range(len(intervals)):
        if i == 0 and intervals[i].mark.strip() == '':
            # Handle blank at the very beginning
            intervals[i + 1].minTime = intervals[i].minTime
        elif i == len(intervals) - 1 and intervals[i].mark.strip() == '':
            # Handle blank at the very end
            intervals[i - 1].maxTime = intervals[i].maxTime
        elif intervals[i].mark.strip() == '':
            # Handle blank in the middle
            half_duration = (intervals[i].maxTime - intervals[i].minTime) / 2
            intervals[i - 1].maxTime = round(intervals[i - 1].maxTime + half_duration, 2)
            intervals[i + 1].minTime = intervals[i - 1].maxTime

    intervals = [i for i in intervals if i.mark != ""]

    for i in range(len(intervals)):
        intervals[i].minTime = round(intervals[i].minTime * ratio)
        intervals[i].maxTime = round(intervals[i].maxTime * ratio)
        if i >= 1:
            assert intervals[i].minTime == intervals[i - 1].maxTime

    merge_idx = 0
    while merge_idx < len(intervals):
        current_interval = intervals[merge_idx]

        # Handle interval at the very beginning
        if merge_idx == 0 and merge_idx < len(intervals) - 1:
            next_interval = intervals[merge_idx + 1]
            if current_interval.maxTime == current_interval.minTime:
                current_interval.maxTime = next_interval.maxTime
                current_interval.mark += " " + next_interval.mark
                intervals.pop(merge_idx + 1)
                continue

        # Handle interval at the very end
        elif merge_idx == len(intervals) - 1 and merge_idx > 0:
            prev_interval = intervals[merge_idx - 1]
            if current_interval.maxTime == current_interval.minTime:
                prev_interval.maxTime = current_interval.maxTime
                prev_interval.mark += " " + current_interval.mark
                intervals.pop(merge_idx)
                continue

        # Handle interval in the middle
        elif merge_idx > 0 and merge_idx < len(intervals) - 1:
            prev_interval = intervals[merge_idx - 1]
            next_interval = intervals[merge_idx + 1]
            if current_interval.maxTime == current_interval.minTime:
                if next_interval.maxTime != next_interval.minTime:
                    prev_interval.maxTime = next_interval.maxTime
                    prev_interval.mark += " " + current_interval.mark + " " + next_interval.mark
                    intervals.pop(merge_idx)  # Remove current interval
                    intervals.pop(merge_idx)  # Remove next interval (same index after previous removal)
                    continue
                else:
                    prev_interval.maxTime = current_interval.maxTime
                    prev_interval.mark += " " + current_interval.mark
                    intervals.pop(merge_idx)
                    continue

        merge_idx += 1

    text_list = []

    for i in range(len(intervals)):
        text_list.append(intervals[i].mark)
        if i > 0:
            assert intervals[i].minTime == intervals[i - 1].maxTime

    transcript_post = " ".join(text_list)
    assert transcript == transcript_post

    if len(intervals) < 3:
        return None, None

    total_data = []
    total_length = []

    N = intervals[-1].maxTime
    num_segments = N // (50 * SEGMENT_SECOND) + 1

    trunc_normal_dist_list = []

    for segment_idx in range(1, num_segments):
        mean = N / num_segments * segment_idx
        std_dev = N / (num_segments * 2)
        dist_min = mean - N / num_segments
        dist_max = mean + N / num_segments

        # Generate truncated normal distribution
        a, b = (dist_min - mean) / std_dev, (dist_max - mean) / std_dev  # Convert min and max values to z-scores
        trunc_normal_dist_list.append(stats.truncnorm(a, b, loc=mean, scale=std_dev))

    start_idx_list = [i.minTime for i in intervals] + [intervals[-1].maxTime]

    for _ in range(epoch):
        segment_idx_list = [0]
        for trunc_normal_dist in trunc_normal_dist_list:
            segment_idx_list.append(
                min(range(len(start_idx_list)), key=lambda i: abs(start_idx_list[i] - trunc_normal_dist.rvs(1)[0])))
            assert 0 <= segment_idx_list[-1] <= len(intervals)
        segment_idx_list.append(len(intervals))
        segment_idx_list = list(set(segment_idx_list))
        segment_idx_list.sort()

        intervals_list = []
        for cur_idx, next_idx in zip(segment_idx_list[:-1], segment_idx_list[1:]):
            intervals_list.append(intervals[cur_idx:next_idx])

        data = [tokenizer.bos_token_id]
        data_type = ["bos"]

        for interval in intervals_list:
            # text
            if random.random() >= 0.5:
                if data_type[-1] == "unit":
                    data += tokenizer("<|continue|>").input_ids[1:]
                data += tokenizer(" ".join([i.mark for i in interval])).input_ids[1:]
                data_type.append("text")
            # unit
            else:
                if data_type[-1] == "text":
                    data += tokenizer("<|continue|>").input_ids[1:]
                data += sequence[interval[0].minTime:interval[-1].maxTime]
                data_type.append("unit")

            # <|correspond|>
            if random.random() >= 0.5:
                data += tokenizer("<|correspond|>").input_ids[1:]
                if data_type[-1] == "unit":
                    data += tokenizer(" ".join([i.mark for i in interval])).input_ids[1:]
                    data_type.append("text")
                else:
                    data += sequence[interval[0].minTime:interval[-1].maxTime]
                    data_type.append("unit")

        data += [tokenizer.eos_token_id]
        assert all(x < tokenizer.vocab_size + 10000 + 2 for x in data)
        total_length.append(len(data))
        data = " ".join([str(i) for i in data])
        total_data.append(data)
    return total_data, total_length


def get_tokenizer(pretrained_model_name_or_path, model_cache_dir, max_length) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, cache_dir=model_cache_dir
    )

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = max_length
    return tokenizer


def set_seed(seed_val):
    random.seed(seed_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True,
                        help="Name or path of the pre-trained tokenizer to load (e.g., 'mistralai/Mistral-7B-v0.1').")
    parser.add_argument('--model_cache_dir', type=str, required=True,
                        help="Directory to store tokenizer checkpoints and other cache files.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input data directory.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the processed output files.")
    parser.add_argument('--epoch', type=int, default=1, help="Number of training epochs. Default is 1.")
    parser.add_argument('--max_length', type=int, default=8192,
                        help="Maximum sequence length for input processing. Default is 8192.")
    parser.add_argument('--seed', type=int, default=43, help="Random seed for reproducibility. Default is 43.")
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = get_tokenizer(args.pretrained_model_name_or_path, args.model_cache_dir, args.max_length)
    tokenizer.truncation_side = "left"

    # Add Special Token (<|continue|>, <|correspond|>)
    _ = tokenizer.add_special_tokens({'additional_special_tokens': [AddedToken(f"<|continue|>", normalized=False)]})
    _ = tokenizer.add_special_tokens({'additional_special_tokens': [AddedToken(f"<|correspond|>", normalized=False)]})

    for unit_idx in range(NUM_UNIT_TOKENS):
        _ = tokenizer.add_special_tokens(
            {'additional_special_tokens': [AddedToken(f"<|unit{unit_idx}|>", normalized=False)]})

    if tokenizer.pad_token_id is None:
        _ = tokenizer.add_special_tokens({"pad_token": "<pad>"})

    textgrid_list = glob.glob(os.path.join(args.data_path, "**/**/*.TextGrid"))
    textgrid_list.sort()

    p = Pool(os.cpu_count())
    futures = []

    for textgrid_path in textgrid_list:
        futures.append(
            p.apply_async(
                align_text_to_sequence,
                args=[textgrid_path, args.epoch]
            )
        )
    p.close()

    total_list = [[] for _ in range(args.epoch)]
    total_length_list = [[] for _ in range(args.epoch)]

    for f in tqdm(futures):
        total_data, total_length = f.get()
        if total_data is not None:
            for idx, (data, length) in enumerate(zip(total_data, total_length)):
                total_list[idx].append(data)
                total_length_list[idx].append(length)
    p.join()

    for i in range(args.epoch):
        data_list = total_list[i]
        length_list = total_length_list[i]

        os.makedirs(f"{args.output_path}", exist_ok=True)

        with open(os.path.join(args.output_path, f'data_epoch{i+1}.json'), "w") as json_file:
            json.dump(length_list, json_file, indent=4)

        with open(os.path.join(args.output_path, f'data_epoch{i+1}.txt'), "w") as f:
            f.write("\n".join(data_list))
