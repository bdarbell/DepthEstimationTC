#!/usr/bin/env python

# 3rd party imports
import torch
import torchvision
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.models.optical_flow import raft_small

# local imports
from ..configs import DEVICE


def get_raft_model() -> tuple:
    """
    Load and return the Raft model for optical flow estimation.

    Returns:
    - raft (torch.nn.Module): The Raft model for optical flow estimation in evaluation mode.
    - transform_raft (torchvision.transforms.Compose): The transformation pipeline for the Raft model.
    """
    
    # Load Raft model for optical flow estimation
    weights_raft = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=weights_raft, progress=False)

    # Load the transformation pipeline for the Raft model
    transform_raft = weights_raft.transforms()

    # Put the model on the device
    raft.to(DEVICE)

    # Put the model in evaluation mode
    raft.eval()

    return raft, transform_raft


def get_flow(img1: torch.Tensor, img2: torch.Tensor, model: torch.nn.Module, transform: torchvision.transforms.Compose) -> torch.Tensor:
    """
    Estimate the optical flow between two images.

    Parameters:
    - img1 (torch.Tensor): The first image.
    - img2 (torch.Tensor): The second image.
    - model (torch.nn.Module): The model for optical flow estimation.
    - transform (torchvision.transforms.Compose): The transformation pipeline for the model.

    Returns:
    - torch.Tensor: The optical flow between the two images.
    """
    # Transform the images
    img1_batch, img2_batch = transform(img1, img2)

    # Estimate the optical flow
    list_of_flows = model(img1_batch.to(DEVICE), img2_batch.to(DEVICE))

    # Return the final (best) predicted flow
    return list_of_flows[-1]