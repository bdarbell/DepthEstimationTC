#!/usr/bin/env python

# standard imports
from typing import Literal, Optional

# 3rd party imports
import torch

# local imports
from ..configs import DEVICE


def get_midas_model(model_type: Literal['MiDaS_small', 'DPT_Hybrid', 'DPT_Large'] = 'MiDaS_small', get_midas_ref: bool = True, path_weights: Optional[str] = None) -> tuple:
    """
    Load and return the MiDaS model for depth estimation.

    Parameters:
    - model_type (Literal['MiDaS_small', 'DPT_Hybrid', 'DPT_Large']): The type of MiDaS model to load. (default: 'MiDaS_small').
    - get_midas_ref (bool): Whether to load the MiDaS reference model. (default: True).
    - path_weights (Optional[str]): The path to the weights file for the MiDaS model. (default: None)

    Returns:
    - midas (torch.nn.Module): The MiDaS model for depth estimation.
    - transform_midas (torchvision.transforms.Compose): The transformation pipeline for the MiDaS model.
    - midas_ref (Optional[torch.nn.Module]): The MiDaS reference model for depth estimation. Only returned if get_midas_ref is True.
    """
    
    # Load a MiDas model for depth estimation
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    if path_weights is not None:
        midas.load_state_dict(torch.load(path_weights))
    
    # Load a MiDas reference model
    midas_ref = torch.hub.load("intel-isl/MiDaS", model_type) if get_midas_ref else None

    # Load the transformation pipeline for the MiDaS model
    transforms_midas = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform_midas = transforms_midas.small_transform if model_type == 'MiDaS_small' else transforms_midas.dpt_transform

    # Put the models on the device
    midas.to(DEVICE)
    if get_midas_ref:
        midas_ref.to(DEVICE)

    return midas, transform_midas, midas_ref


def get_gt_depth(imgs: torch.Tensor, model_ref: torch.nn.Module, size: tuple) -> torch.Tensor:
    """
    Get the ground truth depth map for a batch of images.

    Parameters:
    - imgs (torch.Tensor): The batch of images.
    - model_ref (torch.nn.Module): The reference model for depth estimation.
    - size (tuple): The desired size of the depth map.

    Returns:
    - torch.Tensor: The ground truth depth map.
    """
    # Perform depth estimation using the reference model
    prediction = model_ref(imgs)

    # Resize the depth map to the original resolution
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=size,
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    return prediction