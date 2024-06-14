import cv2
import numpy as np
import glob
import json

def calibrate_multiple_cameras(image_folder, pattern_size=(8, 6)):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = [[] for _ in range(4)]
    grays = []  # Initialize the grays variable

    images = [sorted(glob.glob(f'{image_folder}/calibration_image_cam{i+1}_*.jpg')) for i in range(4)]
    print(f'Number of images from each camera: {len(images[0])}')
    if not all(len(img_list) == len(images[0]) for img_list in images):
        raise ValueError("The number of images from all cameras must be equal.")
            
    for img_set in zip(*images):
        frames = [cv2.imread(img) for img in img_set]
        gray_set = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        rets = [cv2.findChessboardCorners(gray, pattern_size, None) for gray in gray_set]

        if all([ret[0] for ret in rets]):
            obj_points.append(objp)
            grays = gray_set  # Update grays if the condition is met
            for i, (gray, ret) in enumerate(zip(grays, rets)):
                corners = cv2.cornerSubPix(gray, ret[1], (11, 11), (-1, -1), criteria)
                img_points[i].append(corners)

    if not grays:
        raise ValueError("Failed to find chessboard corners in any of the images.")

    calibration_data = {
        'camera_matrices': [],
        'dist_coeffs': [],
        'rotation_vectors': [],
        'translation_vectors': []
    }

    for i in range(4):
        ret, cam_matrix, dist_coef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points[i], grays[0].shape[::-1], None, None)
        calibration_data['camera_matrices'].append(cam_matrix.tolist())
        calibration_data['dist_coeffs'].append(dist_coef.tolist())
        calibration_data['rotation_vectors'].append([rvec.tolist() for rvec in rvecs])
        calibration_data['translation_vectors'].append([tvec.tolist() for tvec in tvecs])
        print(f"Camera {i+1} matrix:\n", cam_matrix)
        print(f"Camera {i+1} distortion coefficients:\n", dist_coef)
        print(f"Camera {i+1} rotation vectors:\n", rvecs)
        print(f"Camera {i+1} translation vectors:\n", tvecs)

    # Save calibration results to JSON
    with open('multi_camera_calibration.json', 'w') as f:
        json.dump(calibration_data, f, indent=4)

# Example usage
calibrate_multiple_cameras('cameras_calibration_images', pattern_size=(8, 6))
