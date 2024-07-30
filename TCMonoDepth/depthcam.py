#!/usr/bin/python

# standard imports
import argparse
import os
import time

# 3rd party imports
import cv2
import numpy as np
import open3d as o3d
import torch
from torchvision.transforms import Compose

# local imports
from networks import TCSmallNet
from networks.transforms import Resize
from networks.transforms import PrepareForNet


def create_output(vertices, colors, filename):
    """
    Create a point cloud file in PLY format.

    Args:
        vertices (numpy.ndarray): Array of 3D vertices.
        colors (numpy.ndarray): Array of RGB colors corresponding to each vertex.
        filename (str): Name of the output file.

    Returns:
        None
    """
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


def display_inlier_outlier(cloud, ind):
    """
    Display the inlier and outlier points in a point cloud.

    Args:
        cloud (open3d.geometry.PointCloud): The point cloud.
        ind (numpy.ndarray): The indices of the inlier points.

    Returns:
        None
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def display_point_cloud(cloud):
    """
    Display the point cloud using Open3D visualization.

    Args:
        cloud (open3d.geometry.PointCloud): The point cloud.

    Returns:
        None
    """
    # Visualize the point cloud
    o3d.visualization.draw_geometries([cloud])


def filter_pc(cloud):
    """
    Apply filters to remove sparse points and clean edges in a point cloud.

    Args:
        cloud (open3d.geometry.PointCloud): The point cloud.

    Returns:
        pc_filtered (open3d.geometry.PointCloud): The filtered point cloud.
        pc_downsampled (open3d.geometry.PointCloud): The downsampled point cloud.
        ind (numpy.ndarray): The indices of the inlier points.

    """
    # Downsample the point cloud
    pc_downsampled = cloud.uniform_down_sample(every_k_points=5)

    # Remove statistical outliers
    pc_filtered, ind = pc_downsampled.remove_statistical_outlier(20, 2.0)

    return pc_filtered, pc_downsampled, ind

def run(args):
    """
    Run the image to point cloud pipeline.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        None
    """
    print("Initialize")

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Load network
    print("Creating model...")
    model = TCSmallNet(args)

    if os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print("Loading model from " + args.resume)
    else:
        print("Loading model path fail, model path does not exist.")
        exit()

    model.to(device).eval()
    print("Loading model done...")

    transform = Compose([
        Resize(
            256,  # width
            256,  # height
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        PrepareForNet(),
    ])

    # Open up the video capture from a webcam
    cap = cv2.VideoCapture(0)

    img0 = cap.read()

    print("Start capturing...")
    print("Press 'ESC' to compute point cloud")

    while cap.isOpened():
        _, img = cap.read()

        start = time.time()

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = transform({"image": frame})["image"]
        frame = torch.from_numpy(frame).to(device).unsqueeze(0)

        # Predict depth
        with torch.no_grad():
            prediction = model.forward(frame)
            prediction = (torch.nn.functional.interpolate(
                prediction,
                size=np.asarray(img0[1]).shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze())

        depth_map = prediction.cpu().numpy()

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

        depth_map_color = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        img_pc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, f'FPS: {round(fps, 2)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow('Image', img)
        cv2.imshow('Depth Map', depth_map_color)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    ### Generate point cloud from the last captured image ###
    output_file = 'pointCloud.ply'

    # Q matrix - Camera parameters
    h, w = img.shape[:2]
    f = 0.8 * w
    Q = np.array(([1.0, 0.0, 0.0, -0.5 * w],
                  [0.0, -1.0, 0.0, 0.5 * h],
                  [0.0, 0.0, 0.0, -f],
                  [0.0, 0.0, 1.0, 0.0]), dtype=np.float32)

    # Project points into 3D
    points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)

    # Get rid of points with value 0 (i.e no depth). The higher, the shorter the view
    mask_map = depth_map > 0.2

    # Mask colors and points.
    output_points = points_3D[mask_map]
    output_colors = img_pc[mask_map]
    create_output(output_points, output_colors, output_file)

    ### Load and visualize point cloud ###
    points_3D = o3d.io.read_point_cloud("pointCloud.ply")  # Read the point cloud
    DEBUG = True
    pc_filtered, pc_downsampled, ind = filter_pc(points_3D)
    if DEBUG:
        display_inlier_outlier(pc_downsampled, ind)
    else:
        display_point_cloud(pc_filtered)  # Display filtered and downsampled point cloud

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Settings
    parser = argparse.ArgumentParser(description="Image to pointcloud pipeline")

    parser.add_argument('--resume', nargs='?', default='./TCMonoDepth/weights/_ckpt_small.pt.tar', type=str, help='path to checkpoint file')
    parser.add_argument('--input', nargs='?', default='./videos', type=str, help='video root path')
    parser.add_argument('--output', nargs='?', default='./output', type=str, help='path to save output')

    args = parser.parse_args()

    run(args)
