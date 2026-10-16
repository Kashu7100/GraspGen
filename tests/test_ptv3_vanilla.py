# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The pure-PyTorch PTv3 against the spconv one the released checkpoints were trained with.

Needs spconv, torch-scatter and CUDA (skipped otherwise). spconv's own output varies run to run
when points share a voxel (which point represents the voxel is a GPU race), so the checks are:
exact agreement where every voxel holds one point, and spconv's own index pairs where they do not.
"""

import pytest
import torch

spconv = pytest.importorskip("spconv.pytorch")
pytest.importorskip("torch_scatter")
if not torch.cuda.is_available():
    pytest.skip("spconv runs on CUDA", allow_module_level=True)

from grasp_gen.models.model_utils import convert_to_ptv3_pc_format  # noqa: E402
from grasp_gen.models.ptv3.ptv3 import PointTransformerV3  # noqa: E402
from grasp_gen.models.ptv3.ptv3_vanilla import HashSparseConv3d, PointTransformerV3Vanilla  # noqa: E402

DEVICE = "cuda"


def unique_voxel_cloud(n=4096, grid=0.01, seed=0):
    """Points on a box surface, one per voxel of the grid the encoder uses."""
    g = torch.Generator().manual_seed(seed)
    faces = torch.randint(0, 6, (n,), generator=g)
    p = torch.rand(n, 3, generator=g) * torch.tensor([0.06, 0.08, 0.2]) - torch.tensor([0.03, 0.04, 0.1])
    axis, side = faces // 2, faces % 2
    p[torch.arange(n), axis] = torch.tensor([0.03, 0.04, 0.1])[axis] * (2 * side - 1)
    p = p.to(DEVICE)
    voxels = torch.div(p - p.min(0)[0], grid, rounding_mode="trunc").int()
    _, inverse = torch.unique(voxels, dim=0, return_inverse=True)
    first = torch.full((int(inverse.max()) + 1,), n, device=DEVICE).scatter_reduce(
        0, inverse, torch.arange(n, device=DEVICE), "amin"
    )
    return p[first.sort().values][None]


@pytest.mark.parametrize("kernel_size", [3, 5])
def test_the_conv_matches_spconv_on_distinct_voxels(kernel_size):
    torch.manual_seed(0)
    coords = torch.unique(torch.randint(0, 12, (400, 3)), dim=0)
    n, cin, cout = len(coords), 4, 6
    feat = torch.randn(n, cin, device=DEVICE)
    ref = spconv.SubMConv3d(cin, cout, kernel_size, padding=kernel_size // 2, bias=True, indice_key="t").to(DEVICE)
    torch.nn.init.normal_(ref.bias)
    indices = torch.cat([torch.zeros(n, 1, dtype=torch.long), coords], 1).int().to(DEVICE)
    expected = ref(spconv.SparseConvTensor(feat, indices, [16] * 3, 1)).features

    conv = HashSparseConv3d(cin, cout, kernel_size).to(DEVICE)
    conv.weight.data = ref.weight.detach().reshape(cout, -1, cin).permute(1, 2, 0).contiguous()
    conv.bias.data = ref.bias.detach().clone()
    got = conv(feat, coords.to(DEVICE), torch.zeros(n, dtype=torch.long, device=DEVICE))
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_the_conv_reads_shared_voxels_as_spconv_pairs_them():
    torch.manual_seed(0)
    base = torch.randint(0, 6, (40, 3))
    coords = torch.cat([base, base[:20], base[:10]])
    coords = coords[torch.randperm(len(coords))]
    n = len(coords)
    indices = torch.cat([torch.zeros(n, 1, dtype=torch.long), coords], 1).int().to(DEVICE)
    x = spconv.SparseConvTensor(torch.randn(n, 2, device=DEVICE), indices, [8] * 3, 1)
    ref = spconv.SubMConv3d(2, 2, 3, padding=1, bias=False, indice_key="t").to(DEVICE)
    spconv_pairs = ref(x).indice_dict["t"].pair_fwd.cpu()  # (27, n): the input each output reads, or -1

    # The pairs this module uses: a unit feature per point, one-hot over points, read back per offset.
    conv = HashSparseConv3d(n, 27 * n, 3).to(DEVICE)
    weight = torch.zeros(27, n, 27, n, device=DEVICE)
    for k in range(27):
        weight[k, :, k, :] = torch.eye(n, device=DEVICE)
    conv.weight.data = weight.reshape(27, n, 27 * n)
    conv.bias.data.zero_()
    read = conv(torch.eye(n, device=DEVICE), coords.to(DEVICE), torch.zeros(n, dtype=torch.long, device=DEVICE))
    read = read.reshape(n, 27, n)  # [output, offset, input]
    ours = torch.where(read.sum(-1) > 0, read.argmax(-1), torch.full((n, 27), -1, device=DEVICE)).T.cpu()

    # spconv's representative is a GPU race, so a few pairs may name another point of the same voxel.
    agree = (ours == spconv_pairs).float().mean().item()
    assert agree > 0.98, agree
    # Who reads what never differs, only which point of a voxel: -1 entries agree exactly.
    assert torch.equal(ours == -1, spconv_pairs == -1)


@pytest.mark.parametrize("part", ["gen", "dis"])
def test_a_trained_spconv_encoder_loads_and_matches_on_distinct_voxels(part):
    torch.manual_seed(0)
    ref = PointTransformerV3(in_channels=3, enable_flash=False, cls_mode=True).to(DEVICE).eval()
    for p in ref.parameters():  # stand-in for trained weights: anything but the init
        p.data.normal_(0, 0.05)
    van = PointTransformerV3Vanilla(in_channels=3).to(DEVICE).eval()
    van.load_state_dict(ref.state_dict())  # spconv format, converted by the load hook
    ref.shuffle_orders = van.shuffle_orders = False

    data = unique_voxel_cloud(seed=1 if part == "gen" else 2)
    with torch.no_grad():
        expected = ref(convert_to_ptv3_pc_format(data, 0.01))
        got = van(convert_to_ptv3_pc_format(data, 0.01))
    assert ((got - expected).norm() / expected.norm()).item() < 1e-4
