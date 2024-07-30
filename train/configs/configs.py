#!/usr/bin/env python

# 3rd party imports
import torch

# CONFIGS

# Path where the trained model weights will be saved
WEIGHTS_OUT_DIR = "out/weights.pth"

# Directory containing the training data
DATA_DIR = "data/NYUv2"

# Model architecture to be used; options could include MiDaS_small or other supported models
MODEL = "MiDaS_small"

# Setting the device to GPU if available, otherwise defaulting to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")