# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

# Modify a class from: https://github.com/shivammehta25/Matcha-TTS

from abc import ABC
import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from voicebox.model.base import BaseModule
from voicebox.model.networks import Transformer


# Reference: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
class CFM(BaseModule, ABC):
    def __init__(
        self,
        solver,
        sigma_min
    ):
        super().__init__()
        self.solver = solver
        self.sigma_min = sigma_min
        self.estimator = None

    def forward(self, x, mask, x1, lengths):
        # random timestep
        t = torch.rand([x1.shape[0], 1, 1], device=x1.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        loss_mask = (torch.arange(0, x.shape[-1], device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        cond = x1 * mask

        u = x1 - (1 - self.sigma_min) * z

        ue = self.estimator(x, y, cond, t, lengths)

        loss_mask = loss_mask * (1 - mask)
        ue = ue * loss_mask
        u = u * loss_mask
        loss = F.mse_loss(ue, u, reduction="sum") / (
            torch.sum(loss_mask) * u.shape[1]
        )
        return loss

    def sample(self, x, z, cond, cond_lengths, t, gradient_scale, speech_prompt):
        t = t.view(z.shape[0], 1, 1)
        mask = torch.zeros_like(cond)
        prompt_length = 0

        if not speech_prompt:
            mask[:, :, :prompt_length] = 1
            cond = cond * mask

        if gradient_scale > 0:
            x = torch.cat([self.n_tokens * torch.ones_like(x), x], dim=0)
            z = torch.cat([z, z], dim = 0)
            cond = torch.cat([torch.zeros_like(cond), cond], dim=0)
            t = torch.cat([t, t], dim=0)
            cond_lengths = torch.cat([cond_lengths, cond_lengths], dim=0)

        dphi_dt = self.estimator(x, z, cond, t, cond_lengths)

        if gradient_scale > 0:
            dphi_dt_uncon, dphi_dt = torch.chunk(dphi_dt, 2, dim=0)
            dphi_dt = dphi_dt + gradient_scale * (dphi_dt - dphi_dt_uncon)
        return dphi_dt, prompt_length

    def solve_euler(self, x, z, cond, cond_lengths, t_span, gradient_scale, speech_prompt, prompt_lengths):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        sol = []
        t = t

        steps = 1
        while steps <= len(t_span) - 1:
            dphi_dt, prompt_length = self.sample(
                x, z, cond, cond_lengths, t, gradient_scale, speech_prompt
            )

            z = z + dt * dphi_dt
            t = t + dt

            if speech_prompt:
                prompt = torch.randn_like(cond)
                prompt = (1 - (1 - self.sigma_min) * t) * prompt + t * cond
                z[:, :, :prompt_lengths[0]] = prompt[:, :, :prompt_lengths[0]]

            sol.append(z)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1][:, :, prompt_length:]

    def solve_heun(self, x, z, cond, cond_lengths, t_span, gradient_scale, speech_prompt, prompt_lengths):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        sol = []

        steps = 1
        while steps <= len(t_span) - 1:
            dphi_dt, prompt_length = self.sample(
                x, z, cond, cond_lengths, t, gradient_scale, speech_prompt
            )

            z_hat = z + dt * dphi_dt
            t = t + dt

            if speech_prompt:
                prompt = torch.randn_like(cond)
                prompt = (1 - (1 - self.sigma_min) * t) * prompt + t * cond
                z_hat[:, :, :prompt_lengths[0]] = prompt[:, :, :prompt_lengths[0]]

            if steps < len(t_span) - 1:
                dphi_dt_hat, prompt_length = self.sample(
                    x, z_hat, cond, cond_lengths, t, gradient_scale, speech_prompt
                )
                z_hat = z + dt * (dphi_dt + dphi_dt_hat) / 2

                if speech_prompt:
                    prompt = torch.randn_like(cond)
                    prompt = (1 - (1 - self.sigma_min) * t) * prompt + t * cond
                    z_hat[:, :, :prompt_lengths[0]] = prompt[:, :, :prompt_lengths[0]]

            z = z_hat

            sol.append(z)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1][:, :, prompt_length:]

    @torch.no_grad()
    def generate(self, x, cond, cond_lengths, n_timesteps, solver="euler", gradient_scale=0.0, speech_prompt=False, prompt_lengths=None):
        z = torch.randn_like(cond)
        if solver == "heun":
            n_timesteps = (n_timesteps + 1) // 2
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=cond.device).to(z)

        if solver == "euler":
            return self.solve_euler(x, z, cond, cond_lengths, t_span, gradient_scale, speech_prompt, prompt_lengths)
        elif solver == "heun":
            return self.solve_heun(x, z, cond, cond_lengths, t_span, gradient_scale, speech_prompt, prompt_lengths)


class Voicebox(CFM, PyTorchModelHubMixin):
    def __init__(self, n_feats, n_tokens, embedding_dim, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, convpos_width,
                 convpos_groups, convpos_depth, attention_dropout, activation_dropout, hidden_dropout, solver,
                 sigma_min):
        super().__init__(
            solver=solver,
            sigma_min=sigma_min
        )
        self.n_tokens = n_tokens
        self.estimator = Transformer(
            n_feats,
            n_tokens + 1,
            embedding_dim,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
            convpos_width,
            convpos_groups,
            convpos_depth,
            attention_dropout,
            activation_dropout,
            hidden_dropout
        )