# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import librosa
import re
from scipy.io.wavfile import write
from seamless_communication.models.unit_extractor import UnitExtractor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from voicebox.util.model_util import initialize_decoder, reconstruct_speech


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


# Function to define bad words lists for generation filtering based on ID ranges
def generate_bad_words_ids(start, end, exclude=[]):
    bad_words_ids = [[token_id] for token_id in range(start, end)]
    for e in exclude:
        bad_words_ids.pop(e)
    return bad_words_ids


@torch.inference_mode()
def sample(user_path, reference_path, model, unit_extractor, voicebox, vocoder, tokenizer, output_path):
    # Define bad words IDs for filtering in generation
    bad_words_ids_unit2text = generate_bad_words_ids(32000, 42003)
    bad_words_ids_text2text = generate_bad_words_ids(32002, 42003)
    bad_words_ids_text2unit = generate_bad_words_ids(0, 32002, exclude=[28705])

    pattern = re.compile(r"<\|unit(\d+)\|>")

    user_wav, sr = librosa.load(user_path, sr=16000)
    user_unit = ''.join(
        [f'<|unit{i}|>' for i in unit_extractor.predict(torch.FloatTensor(user_wav).to(device), 35 - 1).cpu().tolist()])

    model_inputs = {}
    model_input = default_template(user_unit=user_unit)
    model_inputs["input_ids"] = torch.LongTensor(tokenizer(model_input).input_ids).to(device).unsqueeze(0)
    outputs = model.generate(**model_inputs, max_length=tokenizer.model_max_length, do_sample=True,
                             bad_words_ids=bad_words_ids_unit2text, top_p=1.0, top_k=1, temperature=1.0,
                             eos_token_id=tokenizer("\n").input_ids[-1])
    user_text = strip_exact_multiple(tokenizer.decode(outputs[0]).split("<|correspond|>")[-1], ["\n", " "])

    model_inputs = {}
    model_input = default_template(user_unit=user_unit, user_text=user_text)
    model_inputs["input_ids"] = torch.LongTensor(tokenizer(model_input).input_ids).to(device).unsqueeze(0)
    outputs = model.generate(**model_inputs, max_length=tokenizer.model_max_length, do_sample=True,
                             bad_words_ids=bad_words_ids_text2text, top_p=1.0, top_k=1, temperature=1.0,
                             eos_token_id=tokenizer("<|correspond|>").input_ids[-1])
    agent_text = strip_exact_multiple(tokenizer.decode(outputs[0]).split("\n")[-1], ["\n", " ", "<|correspond|>"])

    model_inputs = {}
    model_input = default_template(user_unit=user_unit, user_text=user_text, agent_text=agent_text)
    model_inputs["input_ids"] = torch.LongTensor(tokenizer(model_input).input_ids).to(device).unsqueeze(0)
    outputs = model.generate(**model_inputs, max_length=tokenizer.model_max_length, do_sample=True,
                             bad_words_ids=bad_words_ids_text2unit, top_p=1.0, top_k=1, temperature=1.0,
                             eos_token_id=28705)
    agent_unit = tokenizer.decode(outputs[0]).split("<|correspond|>")[-1]

    matches = [int(x) for x in pattern.findall(agent_unit)]
    agent_unit = torch.LongTensor(matches).to(device)
    audio = reconstruct_speech(agent_unit, device, reference_path, unit_extractor, voicebox, vocoder, n_timesteps=50)

    write(output_path, vocoder.h.sampling_rate, audio)


if __name__ == "__main__":
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

    # Load unit extractor (speech tokenizer)
    unit_extractor = UnitExtractor("xlsr2_1b_v2",
                                   "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
                                   device=device)

    # Load USDM model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path='naver-ai/USDM-DailyTalk',
        cache_dir=args.model_cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).to(device)
    model = torch.compile(model).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path='naver-ai/USDM-DailyTalk',
        cache_dir=args.model_cache_dir
    )

    try:
        sample(args.input_path, args.reference_path, model, unit_extractor, voicebox, vocoder, tokenizer, args.output_path)
    except Exception as e:
        print(f"Error while sampling: {e}")
