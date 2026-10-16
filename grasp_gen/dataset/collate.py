# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Batch collation, apart from dataset.py so inference does not import the training data stack."""

import torch


def collate_batch_keys(batch):
    if len(batch) < 1:
        return
    batch = {key: [data[key] for data in batch] for key in batch[0]}
    if "task" in batch:
        task = batch.pop("task")
        batch["task_is_pick"] = torch.stack([torch.tensor(t == "pick") for t in task])
        batch["task_is_place"] = torch.stack([torch.tensor(t == "place") for t in task])
    for key in batch:
        if key in [
            "inputs",
            "points",
            "seg",
            "object_inputs",
            "bottom_center",
            "cam_pose",
            "ee_pose",
            "placement_masks",
            "placement_region",
        ]:
            batch[key] = torch.stack(batch[key])
        if key in ["contact_dirs", "approach_dirs", "offsets"]:
            batch[key] = torch.cat(batch[key])
    return batch


def collate(batch):
    initial_batch_size = len(batch)
    batch = [data for data in batch if not data.get("invalid", False)]
    final_batch_size = len(batch)
    return collate_batch_keys(batch)
