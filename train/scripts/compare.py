#!/usr/bin/env python

# 3rd party imports
import cv2
import torch
import matplotlib.pyplot as plt


# File paths
filename1 = r"path\to\file1.ppm"
filename2 = r"path\to\file2.ppm"
weights = r"path\to\weights.pth"

# Model type
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
# Possible models to select: "MiDaS_small", "DPT_Hybrid", "DPT_Large"

# Load MiDaS models
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_custom = torch.hub.load("intel-isl/MiDaS", model_type)

# Device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Move models to device
midas.to(device)
midas.eval()
midas_custom.load_state_dict(torch.load(weights))
midas_custom.to(device)

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Select transform based on model type
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Read and convert images
img1 = cv2.imread(filename1)
img2 = cv2.imread(filename2)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Apply transforms to input images
input_batch1 = transform(img1).to(device)
input_batch2 = transform(img2).to(device)

# Perform inference
with torch.no_grad():
    # Run images through MiDaS models
    prediction1 = midas(input_batch1)
    prediction_custom1 = midas_custom(input_batch1)
    prediction2 = midas(input_batch2)
    prediction_custom2 = midas_custom(input_batch2)

    # Resize predictions to match input image size
    prediction1 = torch.nn.functional.interpolate(
        prediction1.unsqueeze(1),
        size=img1.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    prediction_custom1 = torch.nn.functional.interpolate(
        prediction_custom1.unsqueeze(1),
        size=img1.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    prediction2 = torch.nn.functional.interpolate(
        prediction2.unsqueeze(1),
        size=img2.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    prediction_custom2 = torch.nn.functional.interpolate(
        prediction_custom2.unsqueeze(1),
        size=img2.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# Convert predictions to numpy arrays
output1 = prediction1.cpu().numpy()
output_custom1 = prediction_custom1.cpu().numpy()
output2 = prediction2.cpu().numpy()
output_custom2 = prediction_custom2.cpu().numpy()

# Display images and predictions
plt.figure()
f, axarr = plt.subplots(2, 3)
axarr[0, 0].imshow(output1)
axarr[0, 0].set_title('MiDaS (MIX 6)')
axarr[0, 1].imshow(output_custom1)
axarr[0, 1].set_title('MiDaS custom')
axarr[1, 0].imshow(output2)
axarr[1, 1].imshow(output_custom2)
axarr[0, 2].imshow(img1)
axarr[0, 2].set_title('Input')
axarr[1, 2].imshow(img2)

# Remove ticks from axes
for i in range(2):
    for j in range(3):
        axarr[i, j].axes.get_xaxis().set_ticks([])
        axarr[i, j].axes.get_yaxis().set_ticks([])

# Show the plot
plt.show()
