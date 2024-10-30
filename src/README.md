## Instructions for Training

This directory provides instructions for further training a pre-trained LLM (e.g., `Mistral-7B-v0.1`) on speech-text interleaved sequences and adapting it to downstream tasks through supervised fine-tuning. Set up your environment as follows:

```bash
# Step 1: Create and activate a new conda environment
conda create -n usdm python=3.10.15
conda activate usdm

# Step 2: Install common dependencies
conda install -c conda-forge libsndfile=1.0.31
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install .
pip install flash-attn==2.6.3 --no-build-isolation
```
- Environment tested on CUDA V12.4.131, Python 3.10.15, Conda 24.5.0.

---

### Speech-Text Pre-training

First, complete the [data preprocessing](../preprocess/pre-training/README.md) for pre-training. After preprocessing, you’ll have cached training data available. Assuming your data path is `YOUR_DATA_PATH`, you can start pre-training using the following commands:

```bash
accelerate launch \
  --num_processes YOUR_NUM_PROCESSES \
  --num_machines YOUR_NUM_MACHINES \
  --main_process_ip YOUR_MAIN_PROCESS_IP \
  --main_process_port YOUR_MAIN_PROCESS_PORT \
  --machine_rank YOUR_MACHINE_RANK \
  train_pt.py \
  --output_dir YOUR_OUTPUT_DIR \
  --data_path YOUR_DATA_NAME \
  --data_path_cache YOUR_DATA_PATH_CACHE \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --model_cache_dir YOUR_MODEL_CACHE_DIR \
  --train_batch_size YOUR_TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps YOUR_GRADIENT_ACCUMULATION_STEPS \
  --max_input_length 8192 \
  --num_train_epochs 1 \
  --seed SEED \
  --logging_steps YOUR_LOGGING_STEPS \
  --deepspeed_config ../configs/ds_config_zero3_bf16.json \
  --learning_rate 2e-5 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --eval_steps YOUR_EVAL_STEPS \
  --save_steps YOUR_SAVE_STEPS \
  --save_total_limit YOUR_SAVE_TOTAL_LIMIT
```

In our [preprocessing pipeline](../preprocess/pre-training), data for all epochs is preprocessed, shuffled, and then concatenated, so no additional shuffling is required. By setting `num_train_epochs` to 1, you effectively achieve multiple epochs of training, as the preprocessed data includes repeated instances equivalent to the desired number of epochs. Please adjust the number of epochs during preprocessing.

Example: For training on 4 machines, each with 4 GPUs:
```bash
accelerate launch \
  --num_processes 4 \
  --num_machines 4 \
  --main_process_ip YOUR_MAIN_PROCESS_IP \
  --main_process_port YOUR_MAIN_PROCESS_PORT \
  --machine_rank YOUR_MACHINE_RANK (0~3) \
  train_pt.py \
  ...
```

- We used 512 NVIDIA A100 40GB GPUs (64 machines with 8 GPUs each) for pre-training, with a global batch size of 1,024.

#### Key Training Details

`train_pt.py` differs from typical [`transformers`](https://huggingface.co/docs/transformers/v4.46.3/en/index)-based LLM training in two main ways:

1. **CustomMistralForCausalLM**:
   - This custom class enables dynamic packing of multiple interleaved sequences into a single sample up to the model’s max length (8,192 for `Mistral-7B-v0.1`) **without [cross-contamination](https://github.com/huggingface/trl/issues/805#issuecomment-1745166383)**.
   - To use a different LLM backbone, consider the following options:
     1. Implement the custom attention masking we’ve applied for compatibility with your model.
     2. Allow cross-sample interference by setting `train_pt.py`'s trainer to `RandomTrainer` and using any `transformers` LLM in place of `CustomMistralForCausalLM` (note: unverified but simplifies setup).
     3. Avoid packing to prevent interference (at the cost of GPU utilization efficiency).
   - Important: This code is compatible with `transformers` version 4.40.2.
   - Currently, only Flash Attention 2 is supported; other attention mechanisms (e.g., naive, SDPA) are not yet compatible.

2. **SequentialTrainer**:
   - Data is pre-shuffled during preprocessing, and samples are pre-packed close to the max length (8,192) for efficiency. Data is not shuffled during training in `train_pt.py`.
   - To enable shuffling, replace `SequentialTrainer` with `RandomTrainer`.

--- 

### Supervised Fine-tuning

Complete the [data preprocessing](../preprocess/fine-tuning/README.md) for supervised fine-tuning. We provide a [🤗 pre-trained speech-text model](https://huggingface.co/naver-ai/USTM).
Use the following command to fine-tune your model:

```bash
# Fine-tuning all parameters
accelerate launch \
  --num_processes YOUR_NUM_PROCESSES \
  --num_machines YOUR_NUM_MACHINES \
  --main_process_ip YOUR_MAIN_PROCESS_IP \
  --main_process_port YOUR_MAIN_PROCESS_PORT \
  --machine_rank YOUR_MACHINE_RANK \
  train_sft.py \
  --output_dir YOUR_OUTPUT_DIR \
  --data_path_train ../dataset/fine-tuning/dailytalk/preprocessed/train.txt \
  --data_path_test ../dataset/fine-tuning/dailytalk/preprocessed/test.txt \
  --data_path_cache YOUR_DATA_PATH_CACHE \
  --model_name_or_path naver-ai/USTM \
  --model_cache_dir YOUR_MODEL_CACHE_DIR \
  --train_batch_size YOUR_TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps YOUR_GRADIENT_ACCUMULATION_STEPS \
  --max_input_length 8192 \
  --num_train_epochs YOUR_NUM_TRAIN_EPOCHS \
  --seed SEED \
  --logging_steps YOUR_LOGGING_STEPS \
  --deepspeed_config ../configs/ds_config_zero3_bf16.json \
  --learning_rate 2e-5 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --eval_steps YOUR_EVAL_STEPS \
  --save_steps YOUR_SAVE_STEPS \
  --save_total_limit YOUR_SAVE_TOTAL_LIMIT
  
# Fine-tuning with Low-Rank Adaptation (LoRA)
accelerate launch \
  ... \
  train_sft.py \
  ... \
  --lora \
  --lora_r YOUR_LORA_RANK \
  --lora_alpha YOUR_LORA_ALPHA \
  --lora_dropout YOUR_LORA_DROPOUT
```
The code above performs single-turn spoken dialog modeling using DailyTalk, a simple dataset consisting of two speakers. For more diverse tasks, such as handling multiple speakers, multi-turn dialogs, or other applications (e.g., speaker-adaptive TTS, emotion recognition, etc.), **consider preprocessing your custom dataset as needed and fine-tuning the model accordingly.**