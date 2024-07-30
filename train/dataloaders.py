#!/usr/bin/env python

# standard imports
import os

# 3rd party imports
import torch
from torchvision import datasets, transforms

# local imports
from .configs import DATA_DIR


def get_dataloaders(batch_size: int = 16) -> dict:
    """
    Get the dataloaders for the train and test datasets.

    Parameters:
    - batch_size (int): The batch size for the dataloaders. (default: 16)

    Returns:
    - dict: A dictionary containing the train and test dataloaders.
    """
    # Data transform
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop([430,590]),
            transforms.Resize([320,448])])
    
    # Load the dataset
    ds = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), transform=data_transforms) for x in ['train', 'test']}

    # Create the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(ds[x], batch_size=batch_size, shuffle=False) for x in ['train', 'test']}

    return dataloaders