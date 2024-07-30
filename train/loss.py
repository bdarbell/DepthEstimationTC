#!/usr/bin/env python

# 3rd party imports
import torch
import torchvision

# local imports
from .models.depth_estimation import get_gt_depth
from .models.opt_flow_estimation import get_flow
from .utils import warp


def tc_loss(imgs: torch.Tensor, dms: torch.Tensor, imgs_midas: torch.Tensor, raft: torch.nn.Module, transform_raft: torchvision.transforms.Compose, midas_ref: torch.nn.Module, sigma: float = 1000, tc_weight: float = 0.04) -> torch.Tensor:
    """
    Compute the temporal consistency loss.

    Parameters:
    - imgs (torch.Tensor): The input images. Shape: [B, C, H, W].
    - dms (torch.Tensor): The input depth maps. Shape: [B, C, H, W].
    - imgs_midas (torch.Tensor): The input images for MiDaS reference model. Shape: [B, C, H, W].
    - raft (torch.nn.Module): The Raft model for optical flow estimation.
    - transform_raft (torchvision.transforms.Compose): The transformation pipeline for the Raft model.
    - midas_ref (torch.nn.Module): The reference model for depth estimation. (default: None)
    - sigma (float): The sigma value for the occlusion estimation. (default: 1000)
    - tc_weight (float): The weight for the temporal consistency loss. (default: 0.04)

    Returns:
    - torch.Tensor: The total loss.
    """
    # Get the ground truth depth maps
    dm_gt = get_gt_depth(imgs_midas, midas_ref, imgs.shape[2:])[:, None, :, :]

    # Separate images into imgs1 (i) and imgs2 (i+1) (alternating)
    imgs1 = imgs[::2]
    imgs2 = imgs[1::2]
    dm1 = dms[::2]
    dm2 = dms[1::2]
    dm_gt1 = dm_gt[::2]
    dm_gt2 = dm_gt[1::2]

    # Remove the last element if the number of images is odd
    if imgs1.size(0) != imgs2.size(0):
        imgs1 = imgs1[0:imgs1.size(0) - 1, :, :, :]
        dm1 = dm1[0:dm1.size(0) - 1, :, :, :]
        dm_gt1 = dm_gt1[0:dm_gt1.size(0) - 1, :, :, :]

    # Compute optical flow
    flow = get_flow(imgs1, imgs2, raft, transform_raft)

    # Warping: Align pixels of the images and depth maps
    imgs2_warp = warp(imgs2, flow)
    dm2_warp = warp(dm2, flow)

    # Occlusion estimation: Define mask based on pixel colors
    diff = (imgs2_warp - imgs1).float()
    dist = (torch.einsum('bchw, bchw -> bhw', diff, diff) / 3)  # Compute ||img1-img2|| over RGB values
    soft_occlusion_mask = torch.exp(-sigma * dist)

    # Compute depth loss
    dloss = torch.nn.MSELoss()  # l1, l2 or SIloss are possible
    depth_loss = dloss(dm1, dm_gt1) + dloss(dm2, dm_gt2)

    # Compute temporal consistency loss
    tc_loss = ((dm1 - dm2_warp).pow(2) * soft_occlusion_mask).mean()

    # Compute total loss
    loss = depth_loss + tc_weight * tc_loss

    return loss