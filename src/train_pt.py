# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
from dataclasses import dataclass
import datasets
import itertools
import random
import re
import os
from typing import Dict

import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from tokenizers import AddedToken
from peft import get_peft_model, LoraConfig, TaskType

from trainer import SequentialTrainer
from utils import count_parameters, EvalSaveCallback

from model import CustomMistralForCausalLM


IGNORE_INDEX = -100
START_INDEX = 1


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        lengths = [len(instance["input_ids"]) for instance in instances]
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(i) for i in input_ids], batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(i) for i in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        mask = input_ids == 1
        indices = torch.arange(input_ids.shape[1])
        position_ids_torch = torch.ones_like(input_ids) * indices
        indexed_mask = mask.long() * indices
        last_one_positions = torch.cummax(indexed_mask, dim=1).values
        position_ids_torch = position_ids_torch - last_one_positions
        position_ids = [pid.tolist()[:lengths[idx]] for idx, pid in enumerate(position_ids_torch)]

        position_ids_np = np.array(list(itertools.chain.from_iterable(position_ids)))
        subsample_start_idx = torch.LongTensor(np.where(position_ids_np == 0)[0])

        position_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(i) for i in position_ids], batch_first=True, padding_value=START_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            subsample_start_idx=subsample_start_idx
        )


def set_seed(seed_val: int = 42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def main(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision="main",
        cache_dir=args.model_cache_dir
    )

    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = args.max_input_length

    raw_datasets = datasets.load_from_disk(dataset_path=args.data_path)

    model = CustomMistralForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.model_cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False

    for name, param in model.named_parameters():
        # make sure that the parameters are contiguous to train them in distributed settings.
        param.data = param.data.contiguous()
    print(f"Trainable parameters: {count_parameters(model)}")

    # Add new tokens (switch, unit)
    num_new_tokens = 0

    # Add Switch Token (<|switch|>)
    num_new_tokens +=  tokenizer.add_special_tokens(
        {'additional_special_tokens': [AddedToken(f"<|continue|>", normalized=False)]})

    num_new_tokens +=  tokenizer.add_special_tokens(
        {'additional_special_tokens': [AddedToken(f"<|correspond|>", normalized=False)]})

    for unit_idx in range(args.num_unit_tokens):
        num_new_tokens += tokenizer.add_special_tokens(
            {'additional_special_tokens': [AddedToken(f"<|unit{unit_idx}|>", normalized=False)]}
        )

    if tokenizer.pad_token_id is None:
        special_tokens_dict = {
            "pad_token": "<pad>"
        }
        num_new_tokens += tokenizer.add_special_tokens(special_tokens_dict)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.resize_token_embeddings(len(tokenizer))

    std = model.config.initializer_range

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data[len(tokenizer) - num_new_tokens:]
        input_embeddings.normal_(mean=0.0, std=std)

        if model.get_input_embeddings().padding_idx is not None:
            model.get_input_embeddings().weight.data[model.get_input_embeddings().padding_idx].zero_()

        output_embeddings = model.get_output_embeddings().weight.data[len(tokenizer) - num_new_tokens:]
        output_embeddings.normal_(mean=0.0, std=std)

    if args.lora:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 lora_dropout=args.lora_dropout)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.enable_input_require_grads()

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.01,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        report_to=['tensorboard'],
        deepspeed=args.deepspeed_config,
        ddp_timeout=int(os.getenv("DDP_TIMEOUT", 1800))
    )

    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets["test"]
    data_collator = DataCollator(tokenizer=tokenizer)

    trainer_inputs = {"model": model,
                      "args": training_args,
                      "train_dataset": train_dataset,
                      "eval_dataset": test_dataset,
                      "data_collator": data_collator}

    trainer_class = SequentialTrainer
    trainer = trainer_class(**trainer_inputs)

    # To save tokenizer at each checkpoint
    eval_save_callback = EvalSaveCallback(
        args,
        tokenizer,
        test_dataset,
        data_collator,
        trainer.accelerator
    )

    trainer.add_callback(eval_save_callback)

    if args.resume:
        max_global_step = -1
        pattern = r'checkpoint-\d+'

        for root, dirs, files in os.walk(args.output_dir):
            for directory in dirs:
                if re.match(pattern, directory):
                    global_step = int(directory.split('-')[1])
                    if global_step > max_global_step:
                        max_global_step = global_step

        if max_global_step == -1:  # if checkpoint is not found
            print(f'[Warning] There are no checkpoints at the given path: {args.output_dir}')
            trainer.train()
        else:
            print(f'Start training from checkpoint-{max_global_step}...')
            trainer.train(os.path.join(args.output_dir, f'checkpoint-{max_global_step}'))
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model and output files.")
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="Name or path of the pre-trained model to load.")
    parser.add_argument("--model_cache_dir", type=str, default=None, help="Directory to store cached model files.")
    parser.add_argument("--data_path", type=str, default="../dataset/pre-training",
                        help="Path to the pre-training dataset.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Batch size for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of steps for gradient accumulation to increase the effective batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Type of learning rate scheduler (e.g., cosine, linear).")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Ratio of total steps for warmup phase.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for model layers.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Frequency of logging training metrics.")
    parser.add_argument("--max_input_length", type=int, default=8192, help="Maximum input sequence length.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--num_unit_tokens", type=int, default=10000, help="Number of speech tokens used in training.")
    parser.add_argument("--lora", action="store_true",
                        help="Enable LoRA (Low-Rank Adaptation) for model parameter-efficient fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=8, help="Rank parameter for LoRA adaptation.")
    parser.add_argument("--lora_alpha", type=int, default=64, help="Scaling factor for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate applied within LoRA.")
    parser.add_argument("--deepspeed_config", type=str, default="../configs/ds_config_zero3_bf16.json",
                        help="Path to the DeepSpeed configuration file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from a previous checkpoint if available.")
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        help="Evaluation strategy (e.g., steps, epoch).")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between each evaluation.")
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help="Checkpoint save strategy (e.g., steps, epoch).")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between each checkpoint save.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit on the number of saved checkpoints.")
    parser.add_argument("--load_best_model_at_end", action="store_true",
                        help="Load the best model at the end of training based on evaluation metrics.")

    args, _ = parser.parse_known_args()
    print(args.deepspeed_config)
    set_seed(args.seed)
    main(args)
