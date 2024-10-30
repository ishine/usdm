# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

# Modify a class from: https://github.com/jaywalnut310/glow-tts

import numpy as np
import random
import torch
import torchaudio.functional

from voicebox.util.train_util import parse_filelist, load_wav_to_torch
from voicebox.vocoder.meldataset import mel_spectrogram


# Reference: https://github.com/jaywalnut310/glow-tts/blob/master/data_utils.py
class UnitMelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_fft=1024, n_mels=80, sample_rate=22050, hop_length=256, win_length=1024,
                 mean=0.0, std=1.0, f_min=0., f_max=8000, random_seed=None, token_sr=50, **kwargs):
        self.dataset = dataset
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.mean = mean
        self.std = std
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.token_sr = token_sr

        random.seed(random_seed)

    def get_pair(self, data):
        filepath, unit, duration = data['file_path'], data['unit'], data['duration']
        unit, mel = self.get_unit_duration_mel(filepath, unit, duration)
        mel = (mel - self.mean) / self.std
        return (unit, mel)

    def get_unit_duration_mel(self, filepath, unit, duration):
        unit = [int(i) for i in unit.split(" ")]
        duration = [int(i) * round(self.sample_rate / self.token_sr) for i in duration.split(" ")]

        expand_unit = []

        for u, d in zip(unit, duration):
            for _ in range(d):
                expand_unit.append(u)

        new_length = len(expand_unit) // self.hop_length * self.hop_length

        unit = torch.LongTensor(expand_unit)[:new_length].reshape(-1, self.hop_length).mode(1)[0]

        mel = self.get_mel(filepath, new_length)
        assert len(unit) == mel.shape[-1]

        return unit, mel

    def get_mel(self, filepath, length=None):
        audio, sr = load_wav_to_torch(filepath)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)
        if length:
            if length > len(audio):
                audio = torch.nn.functional.pad(audio, [0, length - len(audio)])
            else:
                audio = audio[:length]
        if 'float' in str(audio.dtype):
            audio_norm = audio
        else:
            audio = audio + torch.rand_like(audio)
            audio_norm = audio / 32768.0
        audio_norm = audio_norm.unsqueeze(0)
        mel = mel_spectrogram(audio_norm, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False)
        return mel

    def __getitem__(self, index):
        unit, mel = self.get_pair(self.dataset[int(index)])
        item = {'y': mel, 'x': unit, 'mask': torch.zeros_like(unit)}
        return item

    def __len__(self):
        return len(self.dataset)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class UnitMelBatchCollate(object):
    def __init__(self, out_size, p_uncond, p_drop, r_min, r_max, n_tokens):
        self.out_size = out_size
        self.p_uncond = p_uncond
        self.p_drop = p_drop
        self.r_min = r_min
        self.r_max = r_max
        self.n_tokens = n_tokens

    def __call__(self, batch):
        B = len(batch)
        n_feats = batch[0]['y'].shape[-2]
        # out_size = min(self.out_size, max([i['y'].shape[-1] for i in batch]))
        out_size = self.out_size

        y = torch.zeros((B, n_feats, out_size), dtype=torch.float32)
        x = torch.ones((B, out_size), dtype=torch.long) * self.n_tokens
        # default masking: p_drop (all masking)
        mask = torch.zeros((B, 1, out_size), dtype=torch.long)
        y_lengths = []

        for i, item in enumerate(batch):
            y_, x_, mask_ = item['y'], item['x'], item['mask']

            if out_size is not None:
                if y_.shape[-1] > out_size:
                    max_offset = max(y_.shape[-1] - out_size, 0)
                    out_offset = random.choice(range(0, max_offset))
                    y_ = y_[:, :, out_offset:out_offset + out_size]
                    x_ = x_[out_offset:out_offset + out_size]
                    mask_ = mask_[out_offset:out_offset + out_size]

            y_length = y_.shape[-1]

            # classifier-free training
            if random.random() <= self.p_uncond:
                x_ = torch.ones_like(x_) * self.n_tokens
            # r% masking of sequence
            elif random.random() >= self.p_drop:
                r = random.uniform(self.r_min, self.r_max)
                nonmask_length = max(int(y_length * (1 - r)), 0)
                max_offset = max(y_length - nonmask_length, 0)
                out_offset = random.choice(range(0, max_offset))
                mask_[out_offset:out_offset + nonmask_length] = 1

            y_lengths.append(y_.shape[-1])

            y[i:i+1, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            mask[i, 0, :mask_.shape[-1]] = mask_

        y_lengths = torch.LongTensor(y_lengths)

        return {'x': x, 'mask': mask, 'y': y, 'y_lengths': y_lengths}
