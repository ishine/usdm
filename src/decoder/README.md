## Token-Voicebox

### Overview
This folder contains code for token-Voicebox, which converts the speech tokens outputted by our model, as proposed in [SeamlessM4T](https://github.com/facebookresearch/seamless_communication), into mel-spectrograms. As suggested by the name, token-Voicebox is based on the [Voicebox](https://arxiv.org/abs/2306.15687) architecture and converts 50Hz speech tokens into 87Hz mel-spectrograms. These mel-spectrograms are then converted into 22kHz audio using a pre-trained [BigVGAN v1](https://github.com/NVIDIA/BigVGAN) model (official checkpoint).

Like the original Voicebox, which aims for personalized TTS, token-Voicebox can take a reference audio input for speaker-adaptive synthesis or perform unconditional generation if no reference audio is provided.

The main difference from Voicebox is that the input is a speech token rather than text. This leads to several key distinctions:

1. Unlike text, speech tokens are already **aligned** with the mel-spectrogram being generated, eliminating the need for a duration predictor. Consequently, this code does not include duration predictor training.
   - The original Voicebox trains a duration predictor using the same transformer-based architecture used for mel-spectrogram modeling. If needed, you can use our model architecture to train a duration predictor easily.

2. Speech tokens contain some acoustic information, unlike text. Thus, reference audio serves to supplement this acoustic information rather than provide it entirely. In personalized TTS, this means that providing only reference audio may be insufficient for full personalization; you also need speech tokens that already contain target speaker characteristics.
   - To perform personalized TTS with our model, you will need both the token-Voicebox and a **fine-tuned speech-text model** for generating personalized speech tokens. In this setup, the text-to-token model generates personalized tokens, and token-Voicebox uses these tokens along with reference audio to produce personalized speech.

> **Note:** Set up the environment as described in the USDM setup!

---
### Inference & Speech Reconstruction
You can use the code below to extract tokens from speech and then reconstruct them back into audio. The necessary models will be automatically downloaded to `YOUR_MODEL_CACHE_DIR`.

```python
import librosa
from scipy.io.wavfile import write
from seamless_communication.models.unit_extractor import UnitExtractor
import torch
from voicebox.util.model_util import initialize_decoder, reconstruct_speech

# Model Initialization
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# number of forward pass for flow matching (Voicebox)
n_timesteps = 50

# SeamlessM4T's UnitExtractor
token_extractor = UnitExtractor("xlsr2_1b_v2", "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy", device=device)
# Token Decoder
voicebox, vocoder = initialize_decoder(
   model_cache_dir=YOUR_MODEL_CACHE_DIR,
   device=device
)

# Token Extraction
ground_truth_wav_path = '../../samples/3_0_d1581_user.wav'  # MALE SPEAKER FROM DailyTalk
reference_wav_path = '../../samples/5_1_d1399_ref.wav'  # SAME MALE SPEAKER FROM DailyTalk

ground_truth_audio, ground_truth_sr = librosa.load(ground_truth_wav_path, sr=16000)
ground_truth_token = token_extractor.predict(torch.FloatTensor(ground_truth_audio).to(device), 35 - 1)

# Token-to-Speech Decoder
reconstructed_audio_wo_reference = reconstruct_speech(ground_truth_token, device, None, None, voicebox, vocoder, n_timesteps=n_timesteps)
reconstructed_audio_with_reference = reconstruct_speech(ground_truth_token, device, reference_wav_path, token_extractor, voicebox, vocoder, n_timesteps=n_timesteps)

write("reconstructed_wo_ref.wav", vocoder.h.sampling_rate, reconstructed_audio_wo_reference)
write("reconstructed_with_ref.wav", vocoder.h.sampling_rate, reconstructed_audio_with_reference)
```
---

### Preprocessing

Before training, you need to preprocess your audio data. The dataset should consist of audio files; transcripts are not required.

Assuming your dataset path is `YOUR_DATA_PATH`, dataset name is `YOUR_DATA_NAME`, and split is `YOUR_DATA_SPLIT`, you can run the following command to preprocess data:

```bash
python scripts/preprocess.py --data_path YOUR_DATA_PATH --data_name YOUR_DATA_NAME --data_split YOUR_DATA_SPLIT
```

Running this command will save the preprocessed data in `voicebox/filelists/YOUR_DATA_NAME/YOUR_DATA_SPLIT.txt`.

---

### Training

We provide an example [config file](voicebox/configs/YOUR_DATA_NAME/config.json) for training. We trained the model with a global batch size of 256 on 32 NVIDIA A100-80GB GPUs, using the English subset of Multilingual LibriSpeech and GigaSpeech (about 54k hours). [🤗 Pre-trained token-Voicebox](https://huggingface.co/naver-ai/xlsr-token-Voicebox) is available.

We train Voicebox to generate mel-spectrograms corresponding to 22kHz audio. For faster training and higher GPU efficiency, we recommend resampling your audio to 22kHz before training.

To train the model, run:

```bash
accelerate launch \
  --num_processes NUM_PROCESSES \
  --num_machines NUM_MACHINES \
  --main_process_ip MAIN_PROCESS_IP \
  --main_process_port MAIN_PROCESS_PORT \
  --machine_rank MACHINE_RANK \
  scripts/train.py \
  -c voicebox/configs/YOUR_DATA_NAME/config.json \
  -l YOUR_OUTPUT_PATH \
  -dc YOUR_DATA_CACHE_DIR \
  -mc YOUR_MODEL_CACHE_DIR
```

The standard PyTorch checkpoint, optimizer, and tensorboard logs will be saved to `YOUR_OUTPUT_PATH`. The BigVGAN model required for training will be automatically downloaded to `YOUR_MODEL_CACHE_DIR`, where Hugging Face Hub-compatible Voicebox models in training will also be stored.
