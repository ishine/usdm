# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

# Function names and format from: https://github.com/jaywalnut310/glow-tts/blob/master/train.py

import datasets
import numpy as np
import os
import torch
from torch.cuda import amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from voicebox.model import Voicebox
from voicebox.util.data_util import UnitMelDataset, UnitMelBatchCollate
import voicebox.util.train_util as train_util
from voicebox.vocoder.models import BigVGAN

global_step = 0

def main():
    """Multi Node Multi GPUs Training"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    # Retrieve information from environment variables
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # Initialize distributed process group

    # Set hyperparameters
    hps, config = train_util.get_hparams()

    # Execute training and evaluation on each node
    try:
        train_and_eval(rank, world_size, hps, local_rank)
    except Exception as e:
        print(f"An error occurred in main: {e}")
        dist.destroy_process_group()
        raise


def train_and_eval(rank, world_size, hps, local_rank):
    dist.init_process_group(backend='nccl')

    global global_step

    torch.manual_seed(hps.train.seed)
    np.random.seed(hps.train.seed)

    # Set GPU for each process
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print('Initializing data loaders...')

    raw_dataset = datasets.load_dataset(
        "csv",
        data_files={'train': hps.data.train_filelist_path,
                    'valid': hps.data.valid_filelist_path,
                    'test': hps.data.test_filelist_path
                    },
        delimiter='|',
        column_names=['file_path', 'unit', 'duration'],
        cache_dir=hps.data_cache_dir
    )

    train_dataset = UnitMelDataset(
        dataset=raw_dataset['train'],
        random_seed=hps.train.seed,
        **hps.data
    )
    batch_collate = UnitMelBatchCollate(
        out_size=hps.train.out_size,
        p_uncond=hps.train.p_uncond,
        p_drop=hps.train.p_drop,
        r_min=hps.train.r_min,
        r_max=hps.train.r_max,
        n_tokens=hps.data.n_tokens
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hps.train.batch_size,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=hps.train.num_workers,
        shuffle=False,
        sampler=train_sampler
    )

    if rank == 0:
        val_dataset = UnitMelDataset(
            dataset=raw_dataset['valid'],
            random_seed=hps.train.seed,
            **hps.data
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=hps.train.val_batch_size,
            collate_fn=batch_collate,
            drop_last=True,
            num_workers=hps.train.num_workers,
            shuffle=False
        )

        print('Initializing BigVGAN...')
        vocoder = BigVGAN.from_pretrained(
            pretrained_model_name_or_path="nvidia/bigvgan_22khz_80band",
            cache_dir=hps.model_cache_dir
        ).cuda(rank).eval()
        vocoder.remove_weight_norm()

    scaler = amp.GradScaler(enabled=hps.train.fp16_run)

    if rank == 0:
        print('Initializing Voicebox...')

    model = Voicebox(
        n_feats=hps.data.n_feats,
        n_tokens=hps.data.n_tokens,
        **hps.model
    )
    _ = model.cuda(local_rank)

    if rank == 0:
        nparams = model.nparams / 1e6
        print('Number of model parameters: %.2fm' % (model.nparams / 1e6))
        print('Total parameters: %.2fm' % nparams)

    if rank == 0:
        print('Initializing optimizer...')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=hps.train.learning_rate)

    if rank == 0:
        print("Try loading checkpoint")

    model, load_epoch, optimizer = train_util.load_checkpoint(
        rank,
        hps.model_dir,
        model,
        optimizer=optimizer
    )

    if load_epoch == 0:
        if rank == 0:
            print("Fail loading checkpoint; Training from scratch")

    load_iter = len(train_loader) * load_epoch

    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print('Initializing logger...')
        logger_train = SummaryWriter(log_dir=os.path.join(os.path.join(hps.model_dir, "train"), 'log'))
        logger_valid = SummaryWriter(log_dir=os.path.join(os.path.join(hps.model_dir, "valid"), 'log'))

        print('Logging validation batch...')
        val_batch = val_dataset.sample_test_batch(size=hps.train.test_size)
        with torch.no_grad():
            for i, item in enumerate(val_batch):
                mel = item['y']
                mel = mel * hps.data.std + hps.data.mean
                logger_valid.add_image(f'image_{i}/ground_truth', train_util.plot_tensor(mel.squeeze()), global_step=0, dataformats='HWC')
                audio = vocoder.forward(mel.cuda(local_rank)).cpu().squeeze().clamp(-1, 1).numpy()
                logger_valid.add_audio(f'audio_{i}/ground_truth', audio, global_step, sample_rate=hps.data.sampling_rate)

        print('Start training...')

    global_step = load_iter
    for epoch in range(load_epoch + 1, hps.train.n_epochs + 1):
        try:
            train(rank, local_rank, epoch, hps, model, optimizer, train_loader, logger_train if rank==0 else None, scaler)
            if rank == 0:
                evaluate(rank, local_rank, epoch, hps, model, val_loader, logger_valid)
                synthesize(hps, model, vocoder, val_batch, logger_valid)
                if epoch % hps.train.save_every > 0:
                    continue
                save(epoch, hps, model, optimizer)
        except Exception as e:
            print(f"An error occurred in train_and_eval on rank {rank}: {e}")
            dist.destroy_process_group()
            raise

    dist.destroy_process_group()

def train(rank, local_rank, epoch, hps, model, optimizer, train_loader, logger, scaler):
    if hps.train.fp16_run:
        assert scaler is not None

    train_loader.sampler.set_epoch(epoch)

    global global_step

    if rank == 0:
        print(f'Epoch: {epoch} [iteration: {global_step}]')

    model.train()

    losses = []

    loader = tqdm(train_loader, total=len(train_loader)) if local_rank == 0 else train_loader

    for batch in loader:
        model.zero_grad()
        x = batch['x'].cuda(local_rank, non_blocking=True)
        y, y_lengths = batch['y'].cuda(local_rank, non_blocking=True), batch['y_lengths'].cuda(local_rank, non_blocking=True)
        mask = batch['mask'].cuda(local_rank, non_blocking=True)

        with amp.autocast(enabled=hps.train.fp16_run):
            loss = model(
                x, mask, y, y_lengths
            )

        loss = sum([loss])

        if hps.train.fp16_run:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
            optimizer.step()

        if rank == 0:
            print(f"Loss: {loss.item()}, Grad Norm: {grad_norm}, Step: {global_step}")
            logger.add_scalar('loss/loss', loss.item(), global_step=global_step)
            logger.add_scalar('grad_norm/grad_norm', grad_norm, global_step=global_step)
            losses.append(loss.item())

        global_step += 1


def evaluate(rank, local_rank, epoch, hps, model, val_loader, logger_valid):
    model.eval()
    print('Evaluation...\n')

    losses_valid = []

    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader)):
            x = batch['x'].cuda(local_rank, non_blocking=True)
            y, y_lengths = batch['y'].cuda(local_rank, non_blocking=True), batch['y_lengths'].cuda(local_rank,
                                                                                             non_blocking=True)
            mask = batch['mask'].cuda(local_rank, non_blocking=True)

            with amp.autocast(enabled=hps.train.fp16_run):
                loss = model(
                    x, mask, y, y_lengths
                )

            losses_valid.append(loss.item())

        logger_valid.add_scalar('loss/loss', np.mean(losses_valid), global_step=global_step)

        msg = 'Epoch %d: Validation loss = %.3f ' % (epoch, np.mean(losses_valid))

        with open(f'{hps.model_dir}/train.log', 'a') as f:
            f.write(msg)


def synthesize(hps, model, vocoder, val_batch, logger_valid):
    print('Synthesis...\n')
    model.eval()

    if hasattr(model, 'module'):
        model = model.module

    with torch.no_grad():
        for i, item in enumerate(val_batch):
            x = item['x'].cuda().unsqueeze(0).long()
            y = item['y'].cuda().float()
            y_lengths = torch.LongTensor([y.shape[-1]]).cuda()

            for solver in ["euler", "heun"]:
                for speech_prompt in [False, True]:
                    prompt_lengths = torch.LongTensor([int(y.shape[-1] * 0.3)]).cuda() if speech_prompt else None
                    y_dec = model.generate(
                        x, y, y_lengths, n_timesteps=hps.train.reverse_step, solver=solver, gradient_scale=1.0, speech_prompt=speech_prompt, prompt_lengths=prompt_lengths
                    )

                    y_dec = y_dec * hps.data.std + hps.data.mean
                    audio_dec = vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy()

                    logger_valid.add_audio(f'audio_{i}/solver:{solver}_speech_prompt:{speech_prompt}', audio_dec, global_step, sample_rate=hps.data.sampling_rate)

    print('Done...\n')


def save(epoch, hps, model, optimizer):
    print('Save Checkpoint...\n')

    if hasattr(model, 'module'):
        model = model.module

    # save hf hub model
    model_hf_hub_path = os.path.join(hps.model_cache_dir, "voicebox")
    os.makedirs(model_hf_hub_path, exist_ok=True)
    model.save_pretrained(model_hf_hub_path)

    # save pytorch model
    torch.save({'model': model.state_dict()}, f=f"{hps.model_dir}/voicebox_{epoch}.pt")
    torch.save({'optimizer': optimizer.state_dict()}, f=f"{hps.model_dir}/optimizer_{epoch}.pt")


if __name__ == "__main__":
    main()
