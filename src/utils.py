# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import os
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    PreTrainedTokenizer
)


class EvalSaveCallback(TrainerCallback):
    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer, eval_dataset, data_collator, accelerator):
        self.args = args
        self.tokenizer = tokenizer

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": 0,
            "pin_memory": True,
            "sampler": SequentialSampler(eval_dataset)
        }
        self.eval_dataloader = accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ):
        # some unneeded keyword arguments are given. to ignore and prevent from error, add **kwargs
        # TODO : only rank 0, world 0 instance call this API
        current_ckeckpoint = f"checkpoint-{state.global_step}"
        save_path = os.path.join(args.output_dir, current_ckeckpoint)
        if self.tokenizer is not None:
            print(f'save path: {save_path}')
            self.tokenizer.save_pretrained(save_path)

def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)