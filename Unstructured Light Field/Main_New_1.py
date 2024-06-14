import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

def load_calibration_data(npz_file):
    with np.load(npz_file) as data:
        calibration_data = {
            'camera_matrices': [data['camera_matrices'][i].astype(np.float64) for i in range(data['camera_matrices'].shape[0])],
            'dist_coeffs': [data['dist_coeffs'][i].astype(np.float64) for i in range(data['dist_coeffs'].shape[0])]
        }
    return calibration_data

def load_stereo_params(stereo_params_file):
    with open(stereo_params_file, 'r') as f:
        stereo_params = json.load(f)
    for param in stereo_params:
        param['R'] = np.array(param['R'], dtype=np.float64)
        param['T'] = np.array(param['T'], dtype=np.float64).reshape(3, 1)
    return stereo_params

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
    
    undistorted_images = []
    for cam_idx, cam_images in enumerate(images):
        cam_matrix = calibration_data['camera_matrices'][cam_idx]
        dist_coeffs = calibration_data['dist_coeffs'][cam_idx]
        undistorted_cam_images = [cv2.undistort(img, cam_matrix, dist_coeffs) for img in cam_images]
        undistorted_images.append(undistorted_cam_images)
    
    return undistorted_images, positions

def compute_disparity_map_bm(img_left, img_right):
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Parameters for StereoBM
    numDisparities = 192 # Should be divisible by 16
    blockSize = 5  # Block size must be odd

    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float64) / 16.0
    return disparity

def compute_depth_map(disparity, focal_length, baseline):
    with np.errstate(divide='ignore'):
        depth_map = (focal_length * baseline) / (disparity + 1e-5)
    return depth_map

def rectify_images(img_left, img_right, K1, D1, K2, D2, R, T, img_size):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)
    img_left_rectified = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    img_right_rectified = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)
    return img_left_rectified, img_right_rectified

def compute_disparity_and_depth_for_cameras(images, calibration_data, stereo_params, baseline):
    num_cameras = len(images)
    depth_maps = [[] for _ in range(num_cameras)]

    for i in range(num_cameras):
        for k in range(i+1, num_cameras):
            cam_images_left = images[i]
            cam_images_right = images[k]
            disparities = []
            img_size = cam_images_left[0].shape[:2][::-1]
            for j in range(len(cam_images_left)):
                img_left = cam_images_left[j]
                img_right = cam_images_right[j]
                K1 = calibration_data['camera_matrices'][i]
                D1 = calibration_data['dist_coeffs'][i]
                K2 = calibration_data['camera_matrices'][k]
                D2 = calibration_data['dist_coeffs'][k]
                
                stereo_pair_params = next((param for param in stereo_params if param['camera_pair'] == [i, k]), None)
                print(f'{i} - {k}')
                if stereo_pair_params is None:
                    print(f"Stereo parameters not found for camera pair [{i}, {k}]")
                    continue
                
                R = stereo_pair_params['R']
                T = stereo_pair_params['T']
                
                # Use focal length from the camera matrix (fx from K1)
                focal_length = K1[0, 0]
                
                # Rectify images
                img_left_rectified, img_right_rectified = rectify_images(img_left, img_right, K1, D1, K2, D2, R, T, img_size)

                disparity = compute_disparity_map_bm(img_left_rectified, img_right_rectified)

                # Check if the disparity map is empty or completely black
                if disparity is None or disparity.size == 0 or not np.any(disparity):
                    print(f"Disparity map for camera pair [{i}, {k}] for images {j} is empty or invalid.")
                    continue
                if((i==0 and k==1) or (i==1 and k==2) or (i==2 and k==3)):
                    baseline=52
                elif((i==0 and k==2) or (i==1 and k==3)):
                    baseline=104
                elif(i==0 and k==3):
                    baseline=156
                else:
                    baseline=52
                depth_map = compute_depth_map(disparity, focal_length, baseline)
                depth_maps[i].append(depth_map)
                depth_maps[k].append(depth_map)
    
    # Average depth maps for each camera
    avg_depth_maps = [np.mean(d, axis=0) if d else None for d in depth_maps]
    return avg_depth_maps

def normalize_depth_map(depth_map):
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    normalized_depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    return normalized_depth_map

def refocus_image(image, depth_map, focus_depth, blur_kernel_size=21):
    refocused_image = np.copy(image)
    depth_diff = np.abs(depth_map - focus_depth)
    
    # Blur the entire image
    blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    
    # Create a mask where the focus is sharp
    mask = depth_diff < (depth_diff.max() * 0.1)  # Adjust threshold as needed
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expand mask to 3 dimensions
    
    # Combine blurred and sharp images based on mask
    refocused_image[mask_3d] = image[mask_3d]
    refocused_image[~mask_3d] = blurred_image[~mask_3d]
    
    return refocused_image

def process_images(metadata_path, calibration_file, stereo_params_file, baseline):
    calibration_data = load_calibration_data(calibration_file)
    stereo_params = load_stereo_params(stereo_params_file)
    images, positions = load_images_with_positions(metadata_path, calibration_data)
    
    depth_maps = compute_disparity_and_depth_for_cameras(images, calibration_data, stereo_params, baseline)
    
    return depth_maps, images

# Example usage
metadata_path = 'calibration_images/metadata.json'
calibration_file = 'multi_camera_calibration.npz'
stereo_params_file = 'stereo_params.json'
baseline = 52  # Distance between the cameras in the same units as the focal length (e.g., mm)
focus_depth = 5  # Desired focus depth in the same units as the depth map (e.g., mm)

depth_maps, images = process_images(metadata_path, calibration_file, stereo_params_file, baseline)

# print(len(depth_maps))
# for itwem in depth_maps:
#     plt.imshow(itwem)
#     plt.show()
# Refocus images based on the depth maps
refocused_images = []

# depths = ['depth1.jpg', 'depth2.jpg', 'depth3.jpg', 'depth4.jpg']
# depth_images = [cv2.imread(depth, cv2.IMREAD_GRAYSCALE) for depth in depths]  # Load depth maps as grayscale
# for itwem in depth_images:
#     plt.imshow(itwem)
#     plt.show()
count=0
for depth_map, cam_images in zip(depth_maps, images):
    count+=1
    counter=0
    if depth_map is not None:
        for img in cam_images:
            refocused_img = refocus_image(img, depth_map, focus_depth)
            counter+=1
            refocused_images.append(refocused_img)

# Ensure we have refocused images before trying to combine them
if not refocused_images:
    raise ValueError("No valid refocused images were generated. Please check the previous steps for errors.")

# Combine refocused images
def combine_images(images):
    combined_image = np.zeros_like(images[0], dtype=np.float32)
    alpha = 0.05
    count=0
    for image in images:
        count+=1
        combined_image += image * alpha
        # plt.imshow(combined_image)
        # plt.show()
        # plt.title(f'Image Added - {count} ')
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    return combined_image

combined_refocused_image = combine_images(refocused_images)

# Display the combined refocused image
plt.imshow(cv2.cvtColor(combined_refocused_image, cv2.COLOR_BGR2RGB))
plt.title('Combined Refocused Image')
plt.show()