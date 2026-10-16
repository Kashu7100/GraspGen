# SPDX-FileCopyrightText: Copyright 2025 UC Berkeley Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from diffusers 0.41.0 (src/diffusers/schedulers/scheduling_ddpm.py), Apache License 2.0.
# Only the DDPMScheduler subset this package uses is kept, so diffusers is not a dependency: diffusers
# caps huggingface-hub below 2, which would hold back every project this package is installed into.
# tests/test_ddpm_scheduler.py checks it against diffusers step for step.

"""Denoising diffusion probabilistic model (DDPM) scheduler: the subset the grasp generator uses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DDPMSchedulerOutput:
    """The output of :meth:`DDPMScheduler.step`.

    Attributes:
        prev_sample: The sample at the previous timestep, x_{t-1}.
        pred_original_sample: The predicted denoised sample, x_0.
    """

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    """The betas that discretize the cosine alpha_bar schedule (Nichol & Dhariwal, 2021)."""

    def alpha_bar_fn(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler:
    """diffusers' ``DDPMScheduler`` for epsilon prediction with the fixed-small variance.

    Args:
        num_train_timesteps: Number of diffusion steps the model was trained with.
        beta_start: First beta of a linear schedule.
        beta_end: Last beta of a linear schedule.
        beta_schedule: ``"linear"``, ``"scaled_linear"`` or ``"squaredcos_cap_v2"``.
        clip_sample: Clip the predicted x_0 to ``[-clip_sample_range, clip_sample_range]``.
        prediction_type: Only ``"epsilon"`` (the model predicts the noise).
        clip_sample_range: The clipping bound of ``clip_sample``.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        clip_sample_range: float = 1.0,
    ):
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(
                f"beta_schedule={beta_schedule!r} is not implemented; use 'linear', 'scaled_linear' or "
                f"'squaredcos_cap_v2'."
            )
        if prediction_type != "epsilon":
            raise NotImplementedError(f"prediction_type={prediction_type!r} is not implemented; use 'epsilon'.")

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.prediction_type = prediction_type
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        """Set the timesteps the denoising loop visits: ``num_inference_steps`` of them, "leading" spacing."""
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps={num_inference_steps} cannot exceed num_train_timesteps="
                f"{self.num_train_timesteps}."
            )
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def previous_timestep(self, timestep):
        if self.num_inference_steps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            return torch.tensor(-1) if index == self.timesteps.shape[0] - 1 else self.timesteps[index + 1]
        return timestep - 1

    def _get_variance(self, t) -> torch.Tensor:
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        return torch.clamp(variance, min=1e-20)

    def step(
        self,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> DDPMSchedulerOutput:
        """Reverse one diffusion step: x_t to x_{t-1}, given the predicted noise ``model_output``."""
        t = timestep
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        variance = 0
        if t > 0:
            device = model_output.device
            rand_device = device
            if generator is not None and generator.device.type != device.type and generator.device.type == "cpu":
                rand_device = "cpu"
            variance_noise = torch.randn(
                model_output.shape, generator=generator, device=rand_device, dtype=model_output.dtype
            ).to(device)
            variance = (self._get_variance(t) ** 0.5) * variance_noise

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample + variance, pred_original_sample=pred_original_sample
        )

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor
    ) -> torch.Tensor:
        """Forward diffusion: noise ``original_samples`` to the level of ``timesteps``."""
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = (alphas_cumprod[timesteps] ** 0.5).flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = ((1 - alphas_cumprod[timesteps]) ** 0.5).flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def __len__(self) -> int:
        return self.num_train_timesteps
