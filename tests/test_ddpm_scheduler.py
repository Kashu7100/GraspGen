# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vendored DDPMScheduler against diffusers' own, step for step (skipped without diffusers)."""

import pytest
import torch

from grasp_gen.models.ddpm_scheduler import DDPMScheduler

diffusers_ddpm = pytest.importorskip("diffusers.schedulers.scheduling_ddpm")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
# The schedulers the grasp generator builds: the compositional pair, and the configurable single one.
CONFIGS = [
    dict(beta_schedule="scaled_linear", clip_sample=True),
    dict(beta_schedule="squaredcos_cap_v2", clip_sample=True),
    dict(beta_schedule="linear", clip_sample=False),
]


def pair(num_train_timesteps, **config):
    kwargs = dict(num_train_timesteps=num_train_timesteps, prediction_type="epsilon", **config)
    return DDPMScheduler(**kwargs), diffusers_ddpm.DDPMScheduler(**kwargs)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c["beta_schedule"])
@pytest.mark.parametrize("num_train_timesteps", [10, 1000])
def test_schedule_matches(config, num_train_timesteps):
    ours, theirs = pair(num_train_timesteps, **config)
    assert torch.equal(ours.betas, theirs.betas)
    assert torch.equal(ours.alphas_cumprod, theirs.alphas_cumprod)
    assert torch.equal(ours.timesteps, theirs.timesteps)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c["beta_schedule"])
@pytest.mark.parametrize("set_timesteps", [True, False])
def test_denoising_loop_matches(device, config, set_timesteps):
    ours, theirs = pair(10, **config)
    if set_timesteps:
        ours.set_timesteps(10)
        theirs.set_timesteps(10)
        assert torch.equal(ours.timesteps, theirs.timesteps)
    torch.manual_seed(0)
    sample = torch.randn(64, 6, device=device)
    eps = [torch.randn(64, 6, device=device) for _ in ours.timesteps]
    results = []
    for scheduler in (ours, theirs):
        torch.manual_seed(1)  # step() draws its variance noise from the global generator
        x = sample.clone()
        trace = []
        for k, t in enumerate(scheduler.timesteps):
            out = scheduler.step(model_output=eps[k], timestep=t, sample=x)
            trace.append((out.prev_sample, out.pred_original_sample))
            x = out.prev_sample
        results.append(trace)
    for (a_prev, a_x0), (b_prev, b_x0) in zip(*results):
        assert torch.equal(a_prev, b_prev)
        assert torch.equal(a_x0, b_x0)


@pytest.mark.parametrize("device", DEVICES)
def test_a_seeded_generator_draws_the_same_noise(device):
    ours, theirs = pair(10, beta_schedule="scaled_linear", clip_sample=True)
    sample, eps = torch.randn(8, 3, device=device), torch.randn(8, 3, device=device)
    outs = [
        s.step(eps, torch.tensor(5), sample, generator=torch.Generator(device).manual_seed(7)).prev_sample
        for s in (ours, theirs)
    ]
    assert torch.equal(*outs)


@pytest.mark.parametrize("device", DEVICES)
def test_add_noise_matches(device):
    ours, theirs = pair(10, beta_schedule="squaredcos_cap_v2", clip_sample=True)
    x, noise = torch.randn(32, 6, device=device), torch.randn(32, 6, device=device)
    t = torch.randint(0, 10, (32,), device=device)
    assert torch.equal(ours.add_noise(x, noise, t), theirs.add_noise(x, noise, t))


def test_unsupported_options_raise():
    with pytest.raises(NotImplementedError, match="beta_schedule"):
        DDPMScheduler(beta_schedule="sigmoid")
    with pytest.raises(NotImplementedError, match="prediction_type"):
        DDPMScheduler(prediction_type="v_prediction")
