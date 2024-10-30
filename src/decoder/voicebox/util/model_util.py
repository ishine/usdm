# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

# Modify a function from: https://github.com/jik876/hifi-gan

from collections import OrderedDict
import librosa
import torch
from torchaudio.transforms import Resample

from voicebox.model import Voicebox
from voicebox.util.train_util import load_wav_to_torch
from voicebox.vocoder.meldataset import mel_spectrogram
from voicebox.vocoder.models import BigVGAN


mel_mean = -5.5419
mel_std = 2.1575


# Reference: https://github.com/jik876/hifi-gan/blob/master/meldataset.py
# Function to convert given audio file to mel-spectrogram
def get_mel(filepath, length=None, hps=None):
    audio, sr = load_wav_to_torch(filepath)
    if sr != hps.sampling_rate:
        audio = Resample(orig_freq=sr, new_freq=hps.sampling_rate)(audio)
    if length:
        audio = audio[:length]

    # Normalize audio if needed
    audio_norm = audio.clamp(-1, 1) if "float" in str(audio.dtype) else (audio + torch.rand_like(audio)) / 32768.0
    audio_norm = audio_norm.unsqueeze(0)

    # Generate mel spectrogram using provided parameters
    mel = mel_spectrogram(audio_norm, hps.n_fft, hps.num_mels, hps.sampling_rate,
                          hps.hop_size, hps.win_size, hps.fmin, hps.fmax, center=False)
    return mel


# Function to remove specific patterns from dictionary keys
def remove_pattern_from_keys(ordered_dict, pattern):
    new_dict = OrderedDict()
    for key, value in ordered_dict.items():
        new_key = key.replace(pattern, "")
        new_dict[new_key] = value
    return new_dict


def process_unit(unit, hps, device):
    unit = torch.repeat_interleave(unit, hps.sampling_rate // 50)
    new_length = len(unit) // hps.hop_size * hps.hop_size
    unit = unit[:new_length].reshape(-1, hps.hop_size).mode(1)[0]
    return unit.unsqueeze(0).to(device), new_length


def initialize_decoder(model_cache_dir, device):
    # Token-Voicebox
    voicebox = Voicebox.from_pretrained(
        pretrained_model_name_or_path="naver-ai/xlsr-token-Voicebox",
        cache_dir=model_cache_dir
    ).to(device).eval()

    vocoder = BigVGAN.from_pretrained(
        pretrained_model_name_or_path="nvidia/bigvgan_22khz_80band",
        cache_dir=model_cache_dir
    ).to(device).eval()
    vocoder.remove_weight_norm()
    return voicebox, vocoder


@torch.inference_mode()
def reconstruct_speech(agent_unit, device, reference_path, token_extractor, voicebox, vocoder, n_timesteps=50):
    agent_unit, _ = process_unit(agent_unit, vocoder.h, device)

    if reference_path:
        # Process reference unit and mel if a reference path is provided
        reference_wav, sr = librosa.load(reference_path, sr=16000)
        reference_unit, new_length = process_unit(torch.LongTensor(
            token_extractor.predict(torch.FloatTensor(reference_wav).to(device), 35 - 1).cpu().tolist()), vocoder.h, device)
        reference_mel = get_mel(reference_path, new_length, vocoder.h).to(device)
        reference_mel = (reference_mel - mel_mean) / mel_std

        dummy_y = torch.zeros(agent_unit.shape[0], vocoder.h.num_mels,
                              reference_unit.shape[-1] + agent_unit.shape[-1]).to(device).float()
        dummy_y[:, :, :reference_unit.shape[-1]] = reference_mel
        dummy_y_lengths = torch.LongTensor([dummy_y.shape[-1]]).to(device)
        prompt_lengths = torch.LongTensor([reference_unit.shape[-1]]).to(device)

        unit = torch.cat([reference_unit, agent_unit], dim=-1).to(device)

        y_dec = voicebox.generate(unit, dummy_y, dummy_y_lengths, n_timesteps=n_timesteps, solver="heun",
                                  gradient_scale=1.0, speech_prompt=True, prompt_lengths=prompt_lengths)
        y_dec = y_dec[:, :, reference_unit.shape[-1]:]
    else:
        dummy_y = torch.zeros(agent_unit.shape[0], vocoder.h.num_mels, agent_unit.shape[-1]).to(device).float()
        dummy_y_lengths = torch.LongTensor([dummy_y.shape[-1]]).to(device)

        y_dec = voicebox.generate(agent_unit, dummy_y, dummy_y_lengths, n_timesteps=n_timesteps, solver="heun",
                                  gradient_scale=1.0, speech_prompt=False)

    # Apply inverse normalization and vocoder to generate raw waveform
    y_dec = y_dec * mel_std + mel_mean
    audio_dec = vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy()
    return audio_dec