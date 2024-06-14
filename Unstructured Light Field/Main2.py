import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def load_calibration_data(npz_file):
    with np.load(npz_file) as data:
        calibration_data = {
            'camera_matrices': data['camera_matrices'],
            'dist_coeffs': data['dist_coeffs']
        }
    return calibration_data

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
        cam_matrix = np.array(calibration_data['camera_matrices'][cam_idx])
        dist_coeffs = np.array(calibration_data['dist_coeffs'][cam_idx])
        undistorted_cam_images = [cv2.undistort(img, cam_matrix, dist_coeffs) for img in cam_images]
        undistorted_images.append(undistorted_cam_images)
    
    return undistorted_images, positions

def compute_depth_from_focus(images):
    focus_measures = []
    for cam_images in images:
        fm = []
        for img in cam_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()  # Variance of Laplacian for focus measure
            fm.append(focus_measure)
        focus_measures.append(np.stack(fm, axis=0))
    
    # Stack focus measures to compute depth maps
    depth_maps = []
    for cam_focus_measures in focus_measures:
        depth_map = np.argmax(cam_focus_measures, axis=0)
        depth_maps.append(depth_map)
    depth_maps = np.array(depth_maps)
    return depth_maps

def interp_unstructured(im, x_shift, y_shift):
    h, w = im.shape[:2]
    grid_x, grid_y = np.arange(w), np.arange(h)
    interpolator = RegularGridInterpolator((grid_y, grid_x), im, bounds_error=False, fill_value=0)
    
    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
    points = np.vstack((y_indices.flatten() + y_shift.flatten(), x_indices.flatten() + x_shift.flatten())).T
    return interpolator(points).reshape((h, w))

def refocus_images_unstructured(images, depth_maps, positions, focal_depth):
    if not depth_maps.size or depth_maps[0].size == 0:
        raise ValueError("Depth maps are empty or invalid.")
    
    h, w = depth_maps[0].shape
    num_cameras = len(images)
    refocused_image = np.zeros_like(images[0][0], dtype=np.float32)
    num_images_per_camera = len(images[0])
    
    for cam_idx in range(num_cameras):
        depth_map = depth_maps[cam_idx]
        cam_pos = positions[cam_idx]
        depth_difference = depth_map - focal_depth
        for img in images[cam_idx]:
            refocused_image += np.dstack((interp_unstructured(img[:, :, 0], depth_difference + cam_pos[0], depth_difference + cam_pos[1]),
                                          interp_unstructured(img[:, :, 1], depth_difference + cam_pos[0], depth_difference + cam_pos[1]),
                                          interp_unstructured(img[:, :, 2], depth_difference + cam_pos[0], depth_difference + cam_pos[1])))
    refocused_image /= (num_cameras * num_images_per_camera)
    return refocused_image.astype(np.uint8)

# Example usage
metadata_path = 'metadata.json'
calibration_file = 'multi_camera_calibration.npz'
calibration_data = load_calibration_data(calibration_file)
images, positions = load_images_with_positions(metadata_path, calibration_data)

depth_maps = compute_depth_from_focus(images)

# Ensure depth maps are numpy arrays before normalization
for depth in depth_maps:
    print(f"Depth map shape: {depth.shape}, dtype: {depth.dtype}")  # Debugging line
    depth = np.array(depth, dtype=np.float32)
    if depth.size == 0:
        print("Depth map is empty.")
    norm_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    plt.imshow(norm_depth, cmap='plasma')
    plt.colorbar()
    plt.show()

focal_depth = 1.0  # Example value, adjust as needed
refocused_image = refocus_images_unstructured(images, depth_maps, positions, focal_depth)
plt.imshow(refocused_image)
plt.show()
