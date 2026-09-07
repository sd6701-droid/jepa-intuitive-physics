# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import logging
import sys
import os
from functools import lru_cache
from torch.nn.functional import pad

import torch.distributed as dist

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# Single-machine setup: one "cluster" whose dataset root comes from the
# INTPHYS_DATA_ROOT env var (default: /scratch/sd6701/datasets). Override per-dataset
# paths below if your data lives somewhere else.
CLUSTER = "local"
DATA_ROOT = os.environ.get("INTPHYS_DATA_ROOT", "/scratch/sd6701/datasets")
DATA_ROOT = os.path.abspath(DATA_ROOT)

SUPPORTED_CLUSTERS = {
    "cluster": CLUSTER,
}


@lru_cache()
def get_cluster() -> str:
    # Single-cluster setup (see README): always use the local paths below.
    return CLUSTER

# Gets slurm job vars, to launch another job with the same vars
def slurm_account_partition_and_qos(low_pri: bool) -> str:
    account = os.environ.get("SLURM_JOB_ACCOUNT")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    qos = os.environ.get("SLURM_JOB_QOS")
    assert None not in (account, partition, qos), "This function should only be called by a job scheduled by slurm"
    if low_pri:
        qos = "lowest"
    return account, partition, qos


DATASET_PATHS_BY_CLUSTER = {
    CLUSTER: {
        # IntPhys dev set: https://download-intphys.cognitive-ml.fr/dev.tar.gz
        # layout: <root>/IntPhys/dev/O1/<scene>/{1,2,3,4}/scene/*.png + status.json
        'IntPhys-dev-O1': f'{DATA_ROOT}/IntPhys/dev/O1/',
        'IntPhys-dev-O2': f'{DATA_ROOT}/IntPhys/dev/O2/',
        'IntPhys-dev-O3': f'{DATA_ROOT}/IntPhys/dev/O3/',
        'IntPhys-test-O1': f'{DATA_ROOT}/IntPhys/test/O1/',
        'IntPhys-test-O2': f'{DATA_ROOT}/IntPhys/test/O2/',
        'IntPhys-test-O3': f'{DATA_ROOT}/IntPhys/test/O3/',
        # GRASP level 2: layout <root>/GRASP/videos/level2/{P_,IP_}<Property>/<scene>.mp4
        "GRASP-level-2": f"{DATA_ROOT}/GRASP/videos/level2/",
        # InfLevel-lab: layout <root>/inflevel_lab/{continuity,gravity,solidity}/*.mp4
        "InfLevel-lab": f"{DATA_ROOT}/inflevel_lab/",
        "InfLevel-lab-priming": f"{DATA_ROOT}/inflevel_lab/",
    }
}

PROPERTIES_BY_DATASET = {
    'intphys': ["O1","O2","O3"],
    'grasp': ['Collision',
            'Continuity',
            'Gravity',
            'GravityContinuity',
            'GravityInertia',
            'GravityInertia2',
            'GravitySupport',
            'Inertia',
            'Inertia2',
            'ObjectPermanence',
            'ObjectPermanence2',
            'ObjectPermanence3',
            'SolidityContinuity',
            'SolidityContinuity2',
            'Unchangeableness',
            'Unchangeableness2'],
    'grasp_v2': ['Collision',
            'Continuity',
            'Gravity',
            'GravityContinuity',
            'GravityInertia',
            'GravityInertia2',
            'GravitySupport',
            'Inertia',
            'Inertia2',
            'ObjectPermanence',
            'ObjectPermanence2',
            'ObjectPermanence3',
            'SolidityContinuity',
            'SolidityContinuity2',
            'Unchangeableness',
            'Unchangeableness2'],
    'inflevel_lab': ['continuity',
            'solidity',
            'gravity',
            ],
    'inflevel_lab_priming': ['continuity',
            'solidity',
            'gravity',]
}

def get_dataset_path(dataset: str, cluster=None) -> str:
    if cluster is None:
        cluster = get_cluster()

    return DATASET_PATHS_BY_CLUSTER[cluster][dataset]


def get_dataset_paths(datasets: list[str], is_train: bool = True) -> list[str]:
    cluster = get_cluster()
    assert cluster in DATASET_PATHS_BY_CLUSTER, f"No data paths for environment {cluster}!"
    paths = []
    for dataset in datasets:
        if not is_train:
            dataset = f"{dataset}_val"
        try:
            path = get_dataset_path(dataset, cluster)
        except Exception:
            raise Exception(f"Could not find dataset {dataset} for cluster {cluster}")
        paths.append(path)
    logger.info(f"Datapaths {paths}")
    return paths


def get_time_masks(n_timesteps,spatial_size=(16,16),temporal_size=2,spatial_dim=(224,224),temporal_dim=16,as_bool=False):
    assert n_timesteps % temporal_size == 0
    x,y = spatial_dim
    t = temporal_dim
    
    num_patches_spatial = x/spatial_size[0] * x/spatial_size[0]
    num_patches_time = t/temporal_size
    patches_n_timesteps = int(num_patches_spatial*n_timesteps//temporal_size)
    
    patch_idcs = torch.arange(start=0,end=int(num_patches_spatial*num_patches_time),dtype=int)
    if as_bool:
        mask_enc = patch_idcs < patches_n_timesteps
        mask_pred = patch_idcs >= patches_n_timesteps
    
        full_mask = patch_idcs >= 0
    else:
        mask_enc = patch_idcs[:patches_n_timesteps]
        mask_pred = patch_idcs[patches_n_timesteps:]
    
        full_mask = patch_idcs
    
    return mask_enc, mask_pred,full_mask

def get_breaking_points(clip):
    bps = []
    for diff in [clip[0]-clip[1],clip[0]-clip[2],clip[0]-clip[3]]:
        try:
            i = torch.argwhere(diff.sum(2).sum(2).sum(0)!=0)[0,0].item()
        except:
            i = clip.shape[2]
        bps.append(i)
    return bps


def get_matches(bps):
    if np.argmax(bps) == 0:
        return [[0,1],[2,3]]
    elif np.argmax(bps) == 1:
        return [[0,2],[1,3]]
    else:
        return [[0,3],[1,2]]
    
    
def get_action_timestep(matched_clips):
    diff = matched_clips[0] - matched_clips[1]
    return torch.argwhere(diff.sum(2).sum(2).sum(0)!=0)[0,0].item()


def batch_all_gather(x):
    # Single rank, or no process group (see init_distributed's silent fallback):
    # the gather is the identity, and calling into NCCL would raise instead.
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return x
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def pad_tensors(tensors,max_length,length_axis=-1):
    padded_tensors = []
    for t in tensors:
        padding_needed = max_length - t.size(length_axis)
        padded_tensors.append(pad(t, (0, padding_needed)))
    return padded_tensors