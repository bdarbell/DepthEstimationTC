
# Monocular Depth Estimation with Temporal Consistency
This repository contains code and scripts to train a model enforcing temporal consistency on MiDaS, a depth estimation model, as well as to convert images into pointcloud data using TCMonoDepth [[ref](https://github.com/yu-li/TCMonoDepth)]. The repository is organized into two main directories: `train` and `TCMonoDepth`.

More details about the project can be found [here](https://drive.google.com/file/d/1RCffsG6seDwaQRLczE_MWKorwe9404x7/view?usp=drive_link).

## Repository Structure
- `train/`: Contains code to train the model and enforce temporal consistency on MiDaS.
- `TCMonoDepth/`: Contains a copy of the code from the original TCMonoDepth repository along with the script to estimate depth and convert it to a pointcloud.

## Requirements
Required Python packages can be installed via requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage
### <ins>Training the Model</ins>
To train the model and enforce temporal consistency on MiDaS, run the following commands:
```bash
cd DepthEstimationTC
python -m train
```

#### Configs
The configuration file can be found at `train\configs\configs.py`.

- WEIGHTS_OUT_DIR: Specifies the path where the trained model weights will be saved. The default path is out/weights.pth.
- DATA_DIR: Defines the directory containing the training data. The default data directory is data/NYUv2.
- MODEL: Sets the model architecture to be used. In this example, MiDaS_small is chosen, but other supported models can be specified as needed.
- DEVICE: Determines whether the code will run on a GPU (if available) or fall back to the CPU. This is automatically set by checking the availability of CUDA.

### <ins>Pointcloud Conversion Pipeline</ins>
The TCMonoDepth directory includes a script for converting images into pointcloud data based on TCMonoDepth (`depthcam.py`). The script can be configured through various command-line arguments.

To run the script, use the following commands:
```bash
cd DepthEstimationTC
python TCMonoDepth/depthcam.py [OPTIONS]
```

#### Options

`resume` (optional): Path to the checkpoint file to resume from. Default is ./TCMonoDepth/weights/_ckpt.pt.tar.

`input` (optional): Root path to the input videos. Default is ./videos.

`output` (optional): Path to save the output. Default is ./output.


#### Examples
Using the default settings:

```bash
python TCMonoDepth/depthcam.py
```

Specifying a custom checkpoint path:
```bash
python TCMonoDepth/depthcam.py --resume ./custom_path/checkpoint.pt
```

Custom input and output paths:
```bash
python TCMonoDepth/depthcam.py --input ./my_videos --output ./my_output
```

#### Running the Script
Ensure you have the necessary files and directories as per the specified arguments. Run the script using the command line, adjusting the options as needed for your specific use case.

```bash
python TCMonoDepth/depthcam.py --resume ./weights/_ckpt_small.pt.tar --input ./videos --output ./output
```