import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Load calibration data from .npz file
def load_calibration_data(npz_file):
    with np.load(npz_file) as data:
        calibration_data = {
            'camera_matrices': [data['camera_matrices'][i].astype(np.float32) for i in range(data['camera_matrices'].shape[0])],
            'dist_coeffs': [data['dist_coeffs'][i].astype(np.float32) for i in range(data['dist_coeffs'].shape[0])]
        }
    print("Calibration Data Loaded:", calibration_data)
    return calibration_data

# Load stereo parameters from JSON file
def load_stereo_params(stereo_params_file):
    with open(stereo_params_file, 'r') as f:
        stereo_params = json.load(f)
    for param in stereo_params:
        param['R'] = np.array(param['R'], dtype=np.float32)
        param['T'] = np.array(param['T'], dtype=np.float32).reshape(3, 1)
    print("Stereo Parameters Loaded:", stereo_params)
    return stereo_params

# Load images and their positions from metadata file
def load_images_with_positions(metadata_path, calibration_data):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    images = []
    positions = []
    for batch_key, batch_data in metadata.items():
        cam_images = []
        cam_pos = batch_data["position"]
        for img_path in batch_data["images"].values():
            img = cv2.imread(img_path)
            if img is not None:
                cam_images.append(img)
        images.append(cam_images)
        positions.append(cam_pos)
    
    # Undistort images using calibration data
    undistorted_images = []
    for cam_idx, cam_images in enumerate(images):
        cam_matrix = calibration_data['camera_matrices'][cam_idx]
        dist_coeffs = calibration_data['dist_coeffs'][cam_idx]
        undistorted_cam_images = [cv2.undistort(img, cam_matrix, dist_coeffs) for img in cam_images]
        undistorted_images.append(undistorted_cam_images)
    
    return undistorted_images, positions

# Compute disparity map using StereoBM
def compute_disparity_map_bm(img_left, img_right):
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=80, blockSize=7)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return disparity

# Compute depth map from disparity map
def compute_depth_map(disparity, focal_length, baseline):
    depth_map = (focal_length * baseline) / (disparity + 1e-5)
    return depth_map

# Process images to compute disparity and depth maps
def compute_disparity_and_depth_unstructured(images, calibration_data, stereo_params, focal_length, baseline):
    depth_maps = []
    for cam_images in images:
        disparities = []
        for i in range(len(cam_images) - 1):
            img_left = cam_images[i]
            img_right = cam_images[i + 1]
            
            K1 = calibration_data['camera_matrices'][i]
            D1 = calibration_data['dist_coeffs'][i]
            K2 = calibration_data['camera_matrices'][i + 1]
            D2 = calibration_data['dist_coeffs'][i + 1]
            
            stereo_pair_params = next(param for param in stereo_params if param['camera_pair'] == [i, i + 1])
            R = stereo_pair_params['R']
            T = stereo_pair_params['T']
            
            # No rectification, just undistorted images used
            disparity = compute_disparity_map_bm(img_left, img_right)
            disparities.append(disparity)
        
        avg_disparity = np.mean(disparities, axis=0)
        depth_map = compute_depth_map(avg_disparity, focal_length, baseline)
        depth_maps.append(depth_map)
    
    return depth_maps

# Main processing function
def process_images(metadata_path, calibration_file, stereo_params_file, focal_length, baseline):
    calibration_data = load_calibration_data(calibration_file)
    stereo_params = load_stereo_params(stereo_params_file)
    images, positions = load_images_with_positions(metadata_path, calibration_data)
    
    depth_maps = compute_disparity_and_depth_unstructured(images, calibration_data, stereo_params, focal_length, baseline)
    
    count = 0
    for depth in depth_maps:
        plt.imshow(depth, cmap='gray')
        plt.show()
        cv2.imwrite(f'Main_5_Stereo_BM_depth{count + 1}.jpg', depth)
        count += 1
    
    return depth_maps

# Example usage
metadata_path = 'calibration_images/metadata.json'
calibration_file = 'multi_camera_calibration.npz'
stereo_params_file = 'stereo_params.json'
focal_length = 64
baseline = 15

depth_maps = process_images(metadata_path, calibration_file, stereo_params_file, focal_length, baseline)
