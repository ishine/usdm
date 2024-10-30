# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
from dataclasses import dataclass
import datasets
import random
import re
import os
from typing import Dict

import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType
from trainer import RandomTrainer
from utils import count_parameters, EvalSaveCallback


IGNORE_INDEX = -100
START_INDEX = 1

def mapping_fn(example):
    input_ids = example["text"]
    labels = [int(i) for i in input_ids.split("|")[-1].split(" ")]
    input_ids = [int(i) for i in input_ids.split("|")[0].split(" ")]
    example["input_ids"] = input_ids
    example["labels"] = labels
    del example["text"]
    return example

@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(i) for i in labels], batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(i) for i in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )


def set_seed(seed_val: int = 42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def main(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.model_cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.model_cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False

    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = args.max_input_length

    # ##############
    # Load datasets
    # #############
    raw_datasets = datasets.load_dataset(
        path="text",
        data_files={
            "train": args.data_path_train,
            "test": args.data_path_test
        },
        cache_dir=args.data_path_cache,
        num_proc=32
    )

    raw_datasets["train"] = raw_datasets["train"].shuffle(seed=args.seed)
    raw_datasets["test"] = raw_datasets["test"].shuffle(seed=args.seed)

    raw_datasets = raw_datasets.map(mapping_fn, num_proc=32)

    for name, param in model.named_parameters():
        # make sure that the parameters are contiguous to train them in distributed settings.
        param.data = param.data.contiguous()
    print(f"Trainable parameters: {count_parameters(model)}")

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

    trainer_class = RandomTrainer
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
    parser.add_argument("--model_name_or_path", type=str, default="naver-ai/USTM",
                        help="Name or path of the pre-trained model to load (default: 'naver-ai/USTM').")
    parser.add_argument("--model_cache_dir", type=str, default=None, help="Directory to store cached model files.")
    parser.add_argument("--data_path_train", type=str, nargs='+', required=True,
                        help="List of paths for training datasets. Multiple files can be specified by separating them with spaces.")
    parser.add_argument("--data_path_test", type=str, nargs='+', required=True,
                        help="List of paths for test datasets. Multiple files can be specified by separating them with spaces.")
    parser.add_argument("--data_path_cache", type=str, required=True,
                        help="Directory to store cached dataset files for efficient access.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Batch size for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps for gradient accumulation to increase the effective batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for training.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Type of learning rate scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Ratio of total steps for warmup phase.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for model layers.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Frequency of logging training metrics.")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input sequence length.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs (default: 1).")
    parser.add_argument("--num_unit_tokens", type=int, default=10000,
                        help="Number of speech tokens used in training.")
    parser.add_argument("--lora", action="store_true",
                        help="Enable LoRA (Low-Rank Adaptation) for model parameter-efficient fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=8, help="Rank parameter for LoRA adaptation.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate applied within LoRA.")
    parser.add_argument("--deepspeed_config", type=str, default="../configs/ds_config_zero3_bf16.json",
                        help="Path to the DeepSpeed configuration file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from a previous checkpoint if available.")
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        help="Evaluation strategy, either 'steps' or 'epoch'.")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Number of steps between each evaluation.")
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help="Checkpoint save strategy, either 'steps' or 'epoch'.")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Number of steps between each checkpoint save.")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Limit on the number of saved checkpoints.")
    parser.add_argument("--load_best_model_at_end", action="store_true",
                        help="Load the best model at the end of training based on evaluation metrics.")

    args, _ = parser.parse_known_args()
    print(args.deepspeed_config)
    set_seed(args.seed)
    main(args)
