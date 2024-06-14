import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Load calibration data
def load_calibration_data(npz_file):
    with np.load(npz_file) as data:
        calibration_data = {
            'camera_matrices': data['camera_matrices'],
            'dist_coeffs': data['dist_coeffs'],
            'R': data['rotation_vectors'],
            'T': data['translation_vectors']
        }
    return calibration_data

# Load images and positions from metadata
def load_images(metadata_path):
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
    
    return images, positions

# Undistort and rectify images
def rectify_images(images, calibration_data):
    num_cameras = len(images)
    rectified_images = []
    
    for i in range(num_cameras - 1):
        for j in range(i + 1, num_cameras):
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                calibration_data['camera_matrices'][i], calibration_data['dist_coeffs'][i],
                calibration_data['camera_matrices'][j], calibration_data['dist_coeffs'][j],
                images[i][0].shape[:2], calibration_data['R'][i, j], calibration_data['T'][i, j])
            
            map1x, map1y = cv2.initUndistortRectifyMap(
                calibration_data['camera_matrices'][i], calibration_data['dist_coeffs'][i], R1, P1, images[i][0].shape[:2], cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(
                calibration_data['camera_matrices'][j], calibration_data['dist_coeffs'][j], R2, P2, images[j][0].shape[:2], cv2.CV_32FC1)
            
            rectified_img1 = [cv2.remap(img, map1x, map1y, cv2.INTER_LINEAR) for img in images[i]]
            rectified_img2 = [cv2.remap(img, map2x, map2y, cv2.INTER_LINEAR) for img in images[j]]
            
            rectified_images.append((rectified_img1, rectified_img2))
    
    return rectified_images

# Compute disparity map using StereoSGBM
def compute_disparity_map(img_left, img_right):
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=15)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return disparity

# Convert disparity map to depth map
def compute_depth_map(disparity, focal_length, baseline):
    depth_map = (focal_length * baseline) / (disparity + 1e-5)  # Add a small value to avoid division by zero
    return depth_map

# Compute depth maps for all camera pairs
def compute_disparity_and_depth(images, focal_length, baseline):
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

# Interpolate images using calculated shifts
def interp_unstructured(im, x_shift, y_shift):
    h, w = im.shape[:2]
    grid_x, grid_y = np.arange(w), np.arange(h)
    interpolator = RegularGridInterpolator((grid_y, grid_x), im, bounds_error=False, fill_value=0)
    
    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
    points = np.vstack((y_indices.flatten() + y_shift.flatten(), x_indices.flatten() + x_shift.flatten())).T
    return interpolator(points).reshape((h, w))

# Refocus images using depth maps
def refocus_images(images, depth_maps, positions, focal_depth):
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

# Main function to process images
def main():
    metadata_path = 'calibration_images/metadata.json'
    calibration_file = 'multi_camera_calibration.npz'
    calibration_data = load_calibration_data(calibration_file)
    images, positions = load_images(metadata_path)

    focal_length = 64  # Example value, adjust as needed
    baseline = 15      # Example value, adjust as needed

    rectified_images = rectify_images(images, calibration_data)
    depth_maps = compute_disparity_and_depth(rectified_images, focal_length, baseline)

    # Visualize and save depth maps
    for idx, depth_map in enumerate(depth_maps):
        plt.imshow(depth_map, cmap='plasma')
        plt.colorbar()
        plt.title(f'Depth Map {idx + 1}')
        plt.show()
        cv2.imwrite(f'depth_map_{idx + 1}.jpg', depth_map)

    focal_depth = 0.05  # Example value, adjust as needed
    refocused_image = refocus_images(rectified_images, depth_maps, positions, focal_depth)
    plt.imshow(refocused_image)
    plt.title('Refocused Image')
    plt.show()
    cv2.imwrite('refocused_image.jpg', refocused_image)

if __name__ == "__main__":
    main()
