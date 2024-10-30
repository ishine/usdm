# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import librosa
import re
from scipy.io.wavfile import write
from seamless_communication.models.unit_extractor import UnitExtractor
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.models import _MODELS
from voicebox.util.model_util import reconstruct_speech, initialize_decoder


# Unified template generator for ASR, T2T, and TTS stages
def default_template(user_unit, user_text=None, agent_text=None):
    template = (
        "Below is a conversation between the user and the agent. Each turn includes the user's speech and its corresponding transcript, "
        "along with the agent's response text and the corresponding speech.\n"
        "\n### User\n"
        f"{user_unit}<|correspond|>"
    )
    if user_text:
        template += f"{user_text}\n### Agent\n"
    if agent_text:
        template += f"{agent_text}<|correspond|>"
    return template


# Function to strip multiple exact patterns from a string
def strip_exact_multiple(text, patterns):
    for pattern in patterns:
        if text.startswith(pattern):
            text = text[len(pattern):]
        if text.endswith(pattern):
            text = text[:-len(pattern)]
    return text


@torch.inference_mode()
def sample(user_path, reference_path, model, unit_extractor, voicebox, vocoder,
           sampling_params_unit2text, sampling_params_text2text, sampling_params_text2unit, output_path):
    user_wav, sr = librosa.load(user_path, sr=16000)
    user_unit = ''.join(
        [f'<|unit{i}|>' for i in unit_extractor.predict(torch.FloatTensor(user_wav).to(device), 35 - 1).cpu().tolist()]
    )

    model_input = default_template(user_unit=user_unit)
    outputs = model.generate([model_input], sampling_params_unit2text)
    user_text = strip_exact_multiple(outputs[0].outputs[0].text, ["\n", " "])

    model_input = default_template(user_unit=user_unit, user_text=user_text)
    outputs = model.generate([model_input], sampling_params_text2text)
    agent_text = strip_exact_multiple(outputs[0].outputs[0].text, ["\n", " ", "<|correspond|>"])

    model_input = default_template(user_unit=user_unit, user_text=user_text, agent_text=agent_text)
    outputs = model.generate([model_input], sampling_params_text2unit)
    agent_unit = strip_exact_multiple(outputs[0].outputs[0].text, ["\n", " "])

    matches = [int(x) for x in pattern.findall(agent_unit)]
    agent_unit = torch.LongTensor(matches).to(device)
    audio = reconstruct_speech(agent_unit, device, reference_path, unit_extractor, voicebox, vocoder,
                               n_timesteps=50)
    write(output_path, vocoder.h.sampling_rate, audio)


# Custom processors to block specific token IDs during generation
def bad_word_processor_unit2text(token_ids, logits):
    logits[32000:42003] = float("-inf")
    return logits


def bad_word_processor_text2text(token_ids, logits):
    logits[32002:42003] = float("-inf")
    return logits


def bad_word_processor_text2unit(token_ids, logits):
    logits[0:28705] = float("-inf")
    logits[28706:32002] = float("-inf")
    return logits


if __name__ == "__main__":
    _MODELS['CustomMistralForCausalLM'] = _MODELS['MistralForCausalLM']

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to the input file containing the speech data to process.")
    parser.add_argument('--reference_path', type=str, default=None,
                        help="Path to the reference audio file for speaker adaptation (optional). If not provided, the model will perform speaker unconditional generation.")
    parser.add_argument('--model_cache_dir', type=str, required=True,
                        help="Directory to store the model checkpoints. Change this to specify a custom path for downloaded checkpoints.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save the spoken response.")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load voicebox, vocoder configuration and checkpoint
    voicebox, vocoder = initialize_decoder(args.model_cache_dir, device)

    unit_extractor = UnitExtractor("xlsr2_1b_v2",
                                   "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
                                   device=device)

    model = LLM(model='naver-ai/USDM-DailyTalk', download_dir=args.model_cache_dir, gpu_memory_utilization=0.7)
    tokenizer = AutoTokenizer.from_pretrained('naver-ai/USDM-DailyTalk', cache_dir=args.model_cache_dir)

    sampling_params_unit2text = SamplingParams(
        max_tokens=tokenizer.model_max_length, top_p=1.0, top_k=1, temperature=1.0,
        stop_token_ids=[tokenizer("\n").input_ids[-1]], logits_processors=[bad_word_processor_unit2text]
    )

    sampling_params_text2text = SamplingParams(
        max_tokens=tokenizer.model_max_length, top_p=1.0, top_k=1, temperature=1.0,
        stop_token_ids=[tokenizer("<|correspond|>").input_ids[-1]], logits_processors=[bad_word_processor_text2text]
    )

    sampling_params_text2unit = SamplingParams(
        max_tokens=tokenizer.model_max_length, top_p=1.0, top_k=1, temperature=1.0,
        stop_token_ids=[28705], logits_processors=[bad_word_processor_text2unit]
    )

    pattern = re.compile(r"<\|unit(\d+)\|>")

    try:
        sample(
            args.input_path, args.reference_path, model, unit_extractor, voicebox, vocoder,
            sampling_params_unit2text, sampling_params_text2text, sampling_params_text2unit, args.output_path
        )
    except Exception as e:
        print(f"Error while sampling: {e}")
