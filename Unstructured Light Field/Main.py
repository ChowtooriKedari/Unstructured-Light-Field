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
    print(calibration_data)
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
            print(img_path)
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

def compute_disparity_map(img_left, img_right):
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=144, blockSize=25)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return disparity

def compute_depth_map(disparity, focal_length, baseline):
    depth_map = (focal_length * baseline) / (disparity + 1e-5)
    return depth_map

def compute_disparity_and_depth_unstructured(images, focal_length, baseline):
    depth_maps = []
    for cam_images in images:
        disparities = []
        for i in range(len(cam_images) - 1):
            img_left = cam_images[i]
            img_right = cam_images[i + 1]
            disparity = compute_disparity_map(img_left, img_right)
            disparities.append(disparity)
        avg_disparity = np.mean(disparities, axis=0)
        depth_map = compute_depth_map(avg_disparity, focal_length, baseline)
        depth_maps.append(depth_map)
    return depth_maps

def compute_light_field(images):
    light_field = []
    for cam_images in images:
        light_field.append(np.array(cam_images))
    return np.array(light_field)

def interp_unstructured(im, x_shift, y_shift):
    h, w = im.shape[:2]
    grid_x, grid_y = np.arange(w), np.arange(h)
    interpolator = RegularGridInterpolator((grid_y, grid_x), im, bounds_error=False, fill_value=0)
    
    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
    points = np.vstack((y_indices.flatten() + y_shift.flatten(), x_indices.flatten() + x_shift.flatten())).T
    return interpolator(points).reshape((h, w))

def refocus_images_unstructured(images, depth_maps, positions, focal_depth):
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

metadata_path = 'calibration_images/metadata.json'
calibration_file = 'multi_camera_calibration.npz'
calibration_data = load_calibration_data(calibration_file)
images, positions = load_images_with_positions(metadata_path, calibration_data)

focal_length = 3.0
baseline = 52

depth_maps = compute_disparity_and_depth_unstructured(images, focal_length, baseline)
count=0
for depth in depth_maps:
    plt.imshow(depth)
    plt.show()
    cv2.imwrite(f'Map1_depth{count+1}.jpg', depth)
    count+=1

focal_depth = 0.5
refocused_image = refocus_images_unstructured(images, depth_maps, positions, focal_depth)
plt.imshow(refocused_image)
plt.show()
cv2.imwrite(f'refocused_image.jpg', refocused_image)
