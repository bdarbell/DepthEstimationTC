#!/usr/bin/env python

# 3rd party imports
import torch
import torch.nn.functional as F

# local imports
from .configs import DEVICE


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp an image/tensor back to img1, according to the optical flow.

    Parameters:
    - img (torch.Tensor): The image/tensor to warp. Shape: [B, C, H, W].
    - flow (torch.Tensor): The optical flow. Shape: [B, 2, H, W].

    Returns:
    - torch.Tensor: The warped image/tensor.
    """
    B, _, H, W = img.size()

    # Create mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if img.is_cuda:
        grid = grid.cuda()

    vgrid = grid + flow

    # Scale grid to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    flow = flow.permute(0, 2, 3, 1)
    img = img.float()

    # Perform grid sampling
    output = F.grid_sample(img, vgrid, align_corners=True)

    # Create mask for valid pixels
    mask = torch.ones(img.size(), requires_grad=True).to(DEVICE)
    mask = F.grid_sample(mask, vgrid, align_corners=True)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask