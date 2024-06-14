# import numpy as np
# import cv2
# import json
# import matplotlib.pyplot as plt

# def load_calibration_data(npz_file):
#     with np.load(npz_file) as data:
#         calibration_data = {
#             'camera_matrices': [data['camera_matrices'][i].astype(np.float32) for i in range(data['camera_matrices'].shape[0])],
#             'dist_coeffs': [data['dist_coeffs'][i].astype(np.float32) for i in range(data['dist_coeffs'].shape[0])]
#         }
#     print("Calibration Data Loaded:", calibration_data)
#     return calibration_data

# def load_stereo_params(stereo_params_file):
#     with open(stereo_params_file, 'r') as f:
#         stereo_params = json.load(f)
#     for param in stereo_params:
#         param['R'] = np.array(param['R'], dtype=np.float32)
#         param['T'] = np.array(param['T'], dtype=np.float32).reshape(3, 1)
#     print("Stereo Parameters Loaded:", stereo_params)
#     return stereo_params

# def load_images_with_positions(metadata_path, calibration_data):
#     with open(metadata_path, 'r') as f:
#         metadata = json.load(f)
    
#     images = []
#     positions = []
#     for batch_key, batch_data in metadata.items():
#         cam_images = []
#         cam_pos = batch_data["position"]
#         for img_path in batch_data["images"].values():
#             img = cv2.imread(img_path)
#             if img is not None:
#                 cam_images.append(img)
#         images.append(cam_images)
#         positions.append(cam_pos)
    
#     undistorted_images = []
#     for cam_idx, cam_images in enumerate(images):
#         cam_matrix = calibration_data['camera_matrices'][cam_idx]
#         dist_coeffs = calibration_data['dist_coeffs'][cam_idx]
#         undistorted_cam_images = [cv2.undistort(img, cam_matrix, dist_coeffs) for img in cam_images]
#         undistorted_images.append(undistorted_cam_images)
    
#     return undistorted_images, positions

# def compute_disparity_map_bm(img_left, img_right):
#     gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
#     gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) 
#     stereo = cv2.StereoBM_create(numDisparities=112, blockSize=17)  # Adjusted numDisparities and blockSize
#     disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
#     return disparity

# def compute_depth_map(disparity, focal_length, baseline):
#     with np.errstate(divide='ignore'):  # Ignore division by zero warnings
#         depth_map = (focal_length * baseline) / (disparity + 1e-5)
#     return depth_map

# def normalize_disparity(disparity):
#     min_disp = disparity.min()
#     max_disp = disparity.max()
#     disp_norm = (disparity - min_disp) / (max_disp - min_disp)
#     return disp_norm

# def compute_disparity_and_depth_unstructured(images, calibration_data, stereo_params, baseline):
#     depth_maps = []
#     for cam_images in images:
#         focal_length=0
#         disparities = []
#         for i in range(len(cam_images) - 1):
#             img_left = cam_images[i]
#             img_right = cam_images[i + 1]
#             K1 = calibration_data['camera_matrices'][i]
#             D1 = calibration_data['dist_coeffs'][i]
#             K2 = calibration_data['camera_matrices'][i + 1]
#             D2 = calibration_data['dist_coeffs'][i + 1]
            
#             stereo_pair_params = next(param for param in stereo_params if param['camera_pair'] == [i, i + 1])
#             R = stereo_pair_params['R']
#             T = stereo_pair_params['T']
            
#             # Use focal length from the camera matrix (fx from K1)
#             focal_length = K1[0, 0]
            
#             # No rectification, just undistorted images used
#             disparity = compute_disparity_map_bm(img_left, img_right)
#             disparities.append(disparity)
            
#             # Normalize disparity for visualization
#             disp_norm = normalize_disparity(disparity)
#             # plt.figure()
#             # plt.title(f"Normalized Disparity Map for Image Pair {i} and {i+1}")
#             # plt.imshow(disp_norm, cmap='gray')
#             # plt.colorbar()
#             # plt.show()
        
#         avg_disparity = np.mean(disparities, axis=0)
#         depth_map = compute_depth_map(avg_disparity, focal_length, baseline)
#         depth_maps.append(depth_map)
    
#     return depth_maps

# def process_images(metadata_path, calibration_file, stereo_params_file, baseline):
#     calibration_data = load_calibration_data(calibration_file)
#     stereo_params = load_stereo_params(stereo_params_file)
#     images, positions = load_images_with_positions(metadata_path, calibration_data)
    
#     depth_maps = compute_disparity_and_depth_unstructured(images, calibration_data, stereo_params, baseline)
    
#     count = 0
#     for depth in depth_maps:
#         plt.imshow(depth, cmap='gray')
#         plt.title(f"Depth Map {count + 1}")
#         plt.colorbar()
#         plt.show()
#         cv2.imwrite(f'depth{count + 1}.jpg', depth)
#         count += 1
    
#     return depth_maps

# # Example usage
# metadata_path = 'calibration_images/metadata.json'
# calibration_file = 'multi_camera_calibration.npz'
# stereo_params_file = 'stereo_params.json'
# baseline = 50  # Distance between the cameras in the same units as the focal length (e.g., mm)

# depth_maps = process_images(metadata_path, calibration_file, stereo_params_file, baseline)


import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

def load_calibration_data(npz_file):
    with np.load(npz_file) as data:
        calibration_data = {
            'camera_matrices': [data['camera_matrices'][i].astype(np.float32) for i in range(data['camera_matrices'].shape[0])],
            'dist_coeffs': [data['dist_coeffs'][i].astype(np.float32) for i in range(data['dist_coeffs'].shape[0])]
        }
    print("Calibration Data Loaded:", calibration_data)
    return calibration_data

def load_stereo_params(stereo_params_file):
    with open(stereo_params_file, 'r') as f:
        stereo_params = json.load(f)
    for param in stereo_params:
        param['R'] = np.array(param['R'], dtype=np.float32)
        param['T'] = np.array(param['T'], dtype=np.float32).reshape(3, 1)
    print("Stereo Parameters Loaded:", stereo_params)
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
    stereo = cv2.StereoBM_create(numDisparities=112, blockSize=17) 
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return disparity

def compute_depth_map(disparity, focal_length, baseline):
    with np.errstate(divide='ignore'):
        depth_map = (focal_length * baseline) / (disparity + 1e-5)
    return depth_map

def normalize_disparity(disparity):
    min_disp = disparity.min()
    max_disp = disparity.max()
    disp_norm = (disparity - min_disp) / (max_disp - min_disp)
    return disp_norm

def rectify_images(img_left, img_right, K1, D1, K2, D2, R, T, img_size):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)
    img_left_rectified = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    img_right_rectified = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)
    return img_left_rectified, img_right_rectified

def compute_disparity_and_depth_unstructured(images, calibration_data, stereo_params, baseline):
    depth_maps = []
    num_cameras = len(images[0])
    for cam_images in images:
        focal_length = 0
        disparities = []
        img_size = cam_images[0].shape[:2][::-1]
        for i in range(num_cameras - 1):
            img_left = cam_images[i]
            img_right = cam_images[i + 1]
            K1 = calibration_data['camera_matrices'][i]
            D1 = calibration_data['dist_coeffs'][i]
            K2 = calibration_data['camera_matrices'][i + 1]
            D2 = calibration_data['dist_coeffs'][i + 1]
            
            stereo_pair_params = next(param for param in stereo_params if param['camera_pair'] == [i, i + 1])
            R = stereo_pair_params['R']
            T = stereo_pair_params['T']
            
            # Use focal length from the camera matrix (fx from K1)
            focal_length = K1[0, 0]
            
            # Rectify images
            img_left_rectified, img_right_rectified = rectify_images(img_left, img_right, K1, D1, K2, D2, R, T, img_size)
            
            disparity = compute_disparity_map_bm(img_left_rectified, img_right_rectified)
            disparities.append(disparity)
            
            # Normalize disparity for visualization
            disp_norm = normalize_disparity(disparity)
            # plt.figure()
            # plt.title(f"Normalized Disparity Map for Image Pair {i} and {i+1}")
            # plt.imshow(disp_norm, cmap='gray')
            # plt.colorbar()
            # plt.show()
        
        avg_disparity = np.mean(disparities, axis=0)
        depth_map = compute_depth_map(avg_disparity, focal_length, baseline)
        depth_maps.append(depth_map)
    
    return depth_maps

def process_images(metadata_path, calibration_file, stereo_params_file, baseline):
    calibration_data = load_calibration_data(calibration_file)
    stereo_params = load_stereo_params(stereo_params_file)
    images, positions = load_images_with_positions(metadata_path, calibration_data)
    
    depth_maps = compute_disparity_and_depth_unstructured(images, calibration_data, stereo_params, baseline)
    
    count = 0
    for depth in depth_maps:
        plt.imshow(depth, cmap='gray')
        plt.title(f"Depth Map {count + 1}")
        plt.colorbar()
        plt.show()
        cv2.imwrite(f'depth{count + 1}.jpg', depth)
        count += 1
    
    return depth_maps

# Example usage
metadata_path = 'calibration_images/metadata.json'
calibration_file = 'multi_camera_calibration.npz'
stereo_params_file = 'stereo_params.json'
baseline = 50  # Distance between the cameras in the same units as the focal length (e.g., mm)

depth_maps = process_images(metadata_path, calibration_file, stereo_params_file, baseline)
