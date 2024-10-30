# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import io
import os
import librosa
import numpy as np
import re
import torch
import soundfile as sf
import streamlit as st
from transformers import AutoTokenizer
import wave
from vllm import LLM, SamplingParams
from vllm.model_executor.models import _MODELS
from voicebox.util.model_util import initialize_decoder, reconstruct_speech
from voicebox.vocoder.meldataset import mel_spectrogram


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


# Functions to prevent certain words from being generated
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


# Load necessary models for dialog processing
def load_models():
    from seamless_communication.models.unit_extractor import UnitExtractor

    device = torch.device("cuda")

    st.markdown("Loading Pre-trained Models...")

    # Load voicebox, vocoder configuration and checkpoint
    voicebox, vocoder = initialize_decoder(os.getenv('MODEL_CACHE_DIR'), device)

    usdm = LLM(model='naver-ai/USDM-DailyTalk', download_dir=os.getenv('MODEL_CACHE_DIR'), gpu_memory_utilization=0.7)
    tokenizer = AutoTokenizer.from_pretrained('naver-ai/USDM-DailyTalk', cache_dir=os.getenv('MODEL_CACHE_DIR'))

    # Initialize and load Unit Extractor
    unit_extractor = UnitExtractor("xlsr2_1b_v2",
                                   "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
                                   device=device)

    # Save models to session state for later use
    st.session_state.update({
        'usdm': usdm,
        'tokenizer': tokenizer,
        'voicebox': voicebox,
        'vocoder': vocoder,
        'unit_extractor': unit_extractor,
        'hps': vocoder.h,
        'device': device
    })

    st.markdown("Models Loaded.")


# Strip specific patterns from text if they occur at start or end
def strip_exact(text, pattern):
    if text.startswith(pattern):
        text = text[len(pattern):]
    if text.endswith(pattern):
        text = text[:-len(pattern)]
    return text


# Audio file processing to prepare user input for dialog model
def process_audio(audio_file):
    st.markdown("Processing audio...")
    device = st.session_state['device']

    # Read WAV data and convert to NumPy array
    wav_data = audio_file.read()

    try:
        with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
            frame_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()

            # Set dtype based on sample width
            if sample_width == 2:  # 16-bit audio
                dtype = np.int16
            elif sample_width == 4:  # 32-bit audio
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported audio sample width: {sample_width * 8} bits")
            audio_data = np.frombuffer(wav_data, dtype=dtype) / 32768.
    except:
        # float32
        with io.BytesIO(wav_data) as wav_io:
            audio_data, frame_rate = sf.read(wav_io)

            if np.abs(audio_data).max() > 1:
                audio_data = audio_data / np.abs(audio_data).max()

    # Convert to mono and resample
    audio_data = librosa.to_mono(audio_data)
    audio_data = librosa.resample(y=audio_data, orig_sr=frame_rate, target_sr=16000)

    # Generate unit sequence for model input
    user_unit = ''.join(
        [f'<|unit{i}|>' for i in st.session_state['unit_extractor'].predict(torch.FloatTensor(audio_data).to(device), 34).cpu().tolist()]
    )

    st.markdown("Unit extraction completed.")
    return audio_data, user_unit


# Expand speech tokens to match the length of the mel-spectrogram. (50Hz -> 86.1328Hz (22,050 / 256))
def process_unit(unit, hps, device):
    unit = torch.repeat_interleave(unit, hps.sampling_rate // 50)
    new_length = len(unit) // hps.hop_size * hps.hop_size
    unit = unit[:new_length].reshape(-1, hps.hop_size).mode(1)[0]
    return unit.unsqueeze(0).to(device), new_length


def process_reference_audio(audio_file):
    st.markdown("Processing reference audio...")
    device = st.session_state['device']

    # Read WAV data and convert to NumPy array
    wav_data = audio_file.read()

    try:
        with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
            frame_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()

            # Set dtype based on sample width
            if sample_width == 2:  # 16-bit audio
                dtype = np.int16
            elif sample_width == 4:  # 32-bit audio
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported audio sample width: {sample_width * 8} bits")
            audio_data = np.frombuffer(wav_data, dtype=dtype) / 32768.
    except:
        # float32
        with io.BytesIO(wav_data) as wav_io:
            audio_data, frame_rate = sf.read(wav_io)

            if np.abs(audio_data).max() > 1:
                audio_data = audio_data / np.abs(audio_data).max()

    # Convert to mono and resample
    audio_data = librosa.to_mono(audio_data)
    audio_data_reference = librosa.resample(y=audio_data, orig_sr=frame_rate, target_sr=16000)
    audio_data_voicebox = librosa.resample(y=audio_data, orig_sr=frame_rate, target_sr=st.session_state['hps'].sampling_rate)

    # Generate unit sequence for model input
    reference_unit = torch.LongTensor(st.session_state['unit_extractor'].predict(torch.FloatTensor(audio_data_reference).to(device), 34).cpu().tolist())

    st.markdown("Unit extraction completed.")
    return audio_data_reference, torch.FloatTensor(audio_data_voicebox), reference_unit



def get_mel(audio, length=None, hps=None):
    if length:
        audio = audio[:length]

    # Normalize audio if needed
    audio_norm = audio.clamp(-1, 1) if "float" in str(audio.dtype) else (audio + torch.rand_like(audio)) / 32768.0
    audio_norm = audio_norm.unsqueeze(0)

    # Generate mel spectrogram using provided parameters
    mel = mel_spectrogram(audio_norm, hps.n_fft, hps.num_mels, hps.sampling_rate,
                          hps.hop_size, hps.win_size, hps.fmin, hps.fmax, center=False)
    return mel


# Gather hyperparameters from user input
def get_hparams():
    temperature = st.slider("USDM: Set Temperature", min_value=0.0, max_value=1.0, value=1.0)
    top_p = st.slider("USDM: Set top_p", min_value=0.0, max_value=1.0, value=1.0)
    top_k = st.number_input("USDM: Set top_k", min_value=0, max_value=100, value=1)
    reverse_step = st.number_input("token Voicebox: Set reverse_step for flow matching", min_value=1, max_value=100, value=15)
    return {
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'reverse_step': reverse_step
    }


# Reconstruct speech from unit sequence for agent response
def reconstruct_speech(agent_unit, reference_unit, reference_wav, hparams):
    if reference_wav is not None:
        # Process reference unit and mel if a reference path is provided
        reference_unit, new_length = process_unit(reference_unit, st.session_state['hps'], st.session_state['device'])
        reference_mel = get_mel(reference_wav, new_length, st.session_state['hps']).to(st.session_state['device'])
        reference_mel = (reference_mel - -5.5419) / 2.1575
    else:
        reference_unit = None
        reference_mel = None

    matches = re.compile(r"<\|unit(\d+)\|>").findall(agent_unit)
    matches = [int(x) for x in matches]
    agent_unit = torch.LongTensor(matches).to(st.session_state['device'])
    agent_unit, _ = process_unit(agent_unit, st.session_state['hps'], st.session_state['device'])

    # Use reference mel and unit if provided for speaker-adaptive synthesis
    if reference_unit is not None and reference_mel is not None:
        dummy_y = torch.zeros(agent_unit.shape[0], st.session_state['hps'].num_mels,
                              reference_unit.shape[-1] + agent_unit.shape[-1]).to(st.session_state['device']).float()
        dummy_y[:, :, :reference_unit.shape[-1]] = reference_mel
        dummy_y_lengths = torch.LongTensor([dummy_y.shape[-1]]).to(st.session_state['device'])
        prompt_lengths = torch.LongTensor([reference_unit.shape[-1]]).to(st.session_state['device'])

        unit = torch.cat([reference_unit, agent_unit], dim=-1).to(st.session_state['device'])

        y_dec = st.session_state['voicebox'].generate(unit, dummy_y, dummy_y_lengths, n_timesteps=hparams['reverse_step'], solver="heun",
                                  gradient_scale=1.0, speech_prompt=True, prompt_lengths=prompt_lengths)
        y_dec = y_dec[:, :, reference_unit.shape[-1]:]
    else:
        dummy_y = torch.zeros(agent_unit.shape[0], st.session_state['hps'].num_mels, agent_unit.shape[-1]).to(st.session_state['device']).float()
        dummy_y_lengths = torch.LongTensor([dummy_y.shape[-1]]).to(st.session_state['device'])

        y_dec = st.session_state['voicebox'].generate(agent_unit, dummy_y, dummy_y_lengths, n_timesteps=hparams['reverse_step'], solver="heun",
                                  gradient_scale=1.0, speech_prompt=False)

    # Apply vocoder
    y_dec = y_dec * 2.1575 + -5.5419
    audio_dec = st.session_state['vocoder'].forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy()
    st.audio(audio_dec, format="audio/wav", sample_rate=st.session_state['hps'].sampling_rate)


# Generate and process agent response based on user input
@torch.inference_mode()
def generate_response(user_unit, reference_unit, reference_data, hparams):
    sampling_params_unit2text = SamplingParams(
        max_tokens=st.session_state['tokenizer'].model_max_length, top_p=hparams['top_p'], top_k=hparams['top_k'], temperature=hparams['temperature'],
        stop_token_ids=[st.session_state['tokenizer']("\n").input_ids[-1]], logits_processors=[bad_word_processor_unit2text]
    )

    # Generate user text
    model_input = default_template(user_unit=user_unit)
    outputs = st.session_state['usdm'].generate([model_input], sampling_params_unit2text, use_tqdm=False)
    user_text = strip_exact(strip_exact(outputs[0].outputs[0].text, "\n"), " ")

    # Generate agent text
    sampling_params_text2text = SamplingParams(
        max_tokens=st.session_state['tokenizer'].model_max_length, top_p=hparams['top_p'], top_k=hparams['top_k'], temperature=hparams['temperature'],
        stop_token_ids=[st.session_state['tokenizer']("<|correspond|>").input_ids[-1]], logits_processors=[bad_word_processor_text2text]
    )
    model_input = default_template(user_unit=user_unit, user_text=user_text)
    outputs = st.session_state['usdm'].generate([model_input], sampling_params_text2text, use_tqdm=False)
    agent_text = strip_exact(strip_exact(strip_exact(outputs[0].outputs[0].text, "\n"), " "), "<|correspond|>")

    # Generate agent unit
    sampling_params_text2unit = SamplingParams(
        max_tokens=st.session_state['tokenizer'].model_max_length, top_p=hparams['top_p'], top_k=hparams['top_k'], temperature=hparams['temperature'], stop_token_ids=[28705],
        logits_processors=[bad_word_processor_text2unit]
    )
    model_input = default_template(user_unit=user_unit, user_text=user_text, agent_text=agent_text)
    outputs = st.session_state['usdm'].generate([model_input], sampling_params_text2unit, use_tqdm=False)
    agent_unit = strip_exact(strip_exact(outputs[0].outputs[0].text, "\n"), " ")

    reconstruct_speech(agent_unit, reference_unit, reference_data, hparams)


if __name__ == "__main__":
    _MODELS['CustomMistralForCausalLM'] = _MODELS['MistralForCausalLM']
    st.title("USDM (DailyTalk)")

    # Load models once and keep in session
    if "voicebox" not in st.session_state:
        load_models()

    # Handle file upload
    uploaded_file = st.file_uploader("Upload an Input WAV file for dialog", type=["wav"])
    if uploaded_file is not None:
        audio_data, user_unit = process_audio(uploaded_file)
        st.markdown("User input audio")

        st.audio(audio_data, format="audio/wav", sample_rate=16000)

        reference_file = st.file_uploader("Upload a Reference WAV file to adapt to (Optional)", type=["wav"])
        reference_unit = None
        reference_data_voicebox = None

        if reference_file is not None:
            reference_data, reference_data_voicebox, reference_unit = process_reference_audio(reference_file)
            st.markdown("Reference audio")
            st.audio(reference_data, format="audio/wav", sample_rate=16000)

        # Gather hparams from user input
        hparams = get_hparams()

        # Generate and display response
        if st.button('Generate Response'):
            generate_response(user_unit, reference_unit, reference_data_voicebox, hparams)