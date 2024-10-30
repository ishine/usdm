# Functions from: https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS
# A function from: https://github.com/NVIDIA/tacotron2
# A class and function from: https://github.com/jaywalnut310/glow-tts

import argparse
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchaudio


# Reference: https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/utils.py
def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


# Reference: https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/utils.py
def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


# Reference: https://github.com/NVIDIA/tacotron2/blob/master/utils.py
def load_wav_to_torch(full_path):
    data, sampling_rate = torchaudio.load(full_path)
    return data.squeeze(0), sampling_rate


# Reference: https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/utils.py
def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f if line.strip().split(split_char)[-1] != '']
    return filepaths_and_text


# Reference: https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/utils.py
def latest_checkpoint_path(dir_path, regex="voicebox_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


# Reference: https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/utils.py
def load_checkpoint(rank, logdir, model, optimizer=None):
    if rank == 0:
        print(f'Loading checkpoint from {logdir}...')

    try:
        model_path = latest_checkpoint_path(logdir, regex="voicebox_*.pt")
        model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
        model.load_state_dict(model_dict['model'])

        optim_path = latest_checkpoint_path(logdir, regex="optimizer_*.pt")
        optim_dict = torch.load(optim_path, map_location=lambda loc, storage: loc)
        optimizer.load_state_dict(optim_dict['optimizer'])
        epoch = int(model_path.split('_')[-1].split('.')[0])
    except:
        epoch = 0
        pass

    return model, epoch, optimizer


# Reference: https://github.com/jaywalnut310/glow-tts/blob/13e997689d643410f5d9f1f9a73877ae85e19bc2/utils.py
def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="voicebox/configs/YOUR_DATA_NAME/config.json",
                        help="Path to the JSON configuration file for model training.")
    parser.add_argument('-l', '--log_dir', type=str, default="YOUR_LOG_DIR",
                        help="Directory to save log and checkpoint files.")
    parser.add_argument('-mc', '--model_cache_dir', type=str, default="YOUR_MODEL_CACHE_DIR",
                        help="Directory for caching model (Official BigVGAN, Voicebox in training).")
    parser.add_argument('-dc', '--data_cache_dir', type=str, default="YOUR_DATA_CACHE_DIR",
                        help="Directory for caching data.")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # load config from predefined path (args.config) and save it to config_save_path (in args.log_dir + args.model)
    config_save_path = os.path.join(args.log_dir, "config.json")

    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    with open(config_save_path, "w") as f:
        f.write(data)

    config = json.loads(data)

    hparams = HParams(**config)

    hparams.model_dir = args.log_dir
    hparams.model_cache_dir = args.model_cache_dir
    hparams.data_cache_dir = args.data_cache_dir
    hparams.train.out_size = None if hparams.train.out_size_second is None \
        else hparams.train.out_size_second * hparams.data.sampling_rate // hparams.data.hop_length
    return hparams, config


# Reference: https://github.com/jaywalnut310/glow-tts/blob/13e997689d643410f5d9f1f9a73877ae85e19bc2/utils.py
class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()