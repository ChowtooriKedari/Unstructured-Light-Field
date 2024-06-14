# import cv2
# import numpy as np
# import glob
# import json

# def calibrate_multiple_cameras(image_folder, pattern_size=(8, 6)):
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#     objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
#     objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

#     obj_points = []
#     img_points = [[] for _ in range(4)]
#     grays = []

#     images = [sorted(glob.glob(f'{image_folder}/calibration_image_cam{i+1}_*.jpg')) for i in range(4)]
#     if not all(len(img_list) == len(images[0]) for img_list in images):
#         raise ValueError("The number of images from all cameras must be equal.")
            
#     for img_set in zip(*images):
#         frames = [cv2.imread(img) for img in img_set]
#         gray_set = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
#         rets = [cv2.findChessboardCorners(gray, pattern_size, None) for gray in gray_set]

#         if all([ret[0] for ret in rets]):
#             obj_points.append(objp)
#             grays = gray_set
#             for i, (gray, ret) in enumerate(zip(grays, rets)):
#                 corners = cv2.cornerSubPix(gray, ret[1], (11, 11), (-1, -1), criteria)
#                 img_points[i].append(corners)

#     if not grays:
#         raise ValueError("Failed to find chessboard corners in any of the images.")

#     camera_matrices = []
#     dist_coeffs = []

#     for i in range(4):
#         ret, cam_matrix, dist_coef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points[i], grays[0].shape[::-1], None, None)
#         camera_matrices.append(cam_matrix)
#         dist_coeffs.append(dist_coef)
#         print(f"Camera {i+1} matrix:\n", cam_matrix)
#         print(f"Camera {i+1} distortion coefficients:\n", dist_coef)

#     # Perform stereo calibration for each pair of cameras
#     stereo_params = []
#     for i in range(3):
#         for j in range(i + 1, 4):
#             retval, camera_matrices[i], dist_coeffs[i], camera_matrices[j], dist_coeffs[j], R, T, E, F = cv2.stereoCalibrate(
#                 obj_points, img_points[i], img_points[j], camera_matrices[i], dist_coeffs[i], camera_matrices[j], dist_coeffs[j], grays[0].shape[::-1],
#                 criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
#             stereo_params.append((i, j, R.tolist(), T.tolist()))

#             print(f"Rotation matrix between camera {i+1} and camera {j+1}:\n", R)
#             print(f"Translation vector between camera {i+1} and camera {j+1}:\n", T)

#     # Save calibration results
#     np.savez('multi_camera_calibration.npz', camera_matrices=camera_matrices, dist_coeffs=dist_coeffs)

#     # Convert numpy arrays to lists for JSON serialization
#     serializable_stereo_params = [{'camera_pair': (i, j), 'R': R, 'T': T} for (i, j, R, T) in stereo_params]

#     with open('stereo_params.json', 'w') as f:
#         json.dump(serializable_stereo_params, f, indent=4)

# # Example usage
# calibrate_multiple_cameras('cameras_calibration_images', pattern_size=(8, 6))
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
    grays = []

    images = [sorted(glob.glob(f'{image_folder}/calibration_image_cam{i+1}_*.jpg')) for i in range(4)]
    if not all(len(img_list) == len(images[0]) for img_list in images):
        raise ValueError("The number of images from all cameras must be equal.")
            
    for img_set in zip(*images):
        frames = [cv2.imread(img) for img in img_set]
        gray_set = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        rets = [cv2.findChessboardCorners(gray, pattern_size, None) for gray in gray_set]

        if all([ret[0] for ret in rets]):
            obj_points.append(objp)
            grays = gray_set
            for i, (gray, ret) in enumerate(zip(grays, rets)):
                corners = cv2.cornerSubPix(gray, ret[1], (11, 11), (-1, -1), criteria)
                img_points[i].append(corners)

    if not grays:
        raise ValueError("Failed to find chessboard corners in any of the images.")

    camera_matrices = []
    dist_coeffs = []

    for i in range(4):
        ret, cam_matrix, dist_coef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points[i], grays[0].shape[::-1], None, None)
        camera_matrices.append(cam_matrix.astype(np.float32))
        dist_coeffs.append(dist_coef.astype(np.float32))
        print(f"Camera {i+1} matrix:\n", cam_matrix)
        print(f"Camera {i+1} distortion coefficients:\n", dist_coef)

    # Perform stereo calibration for each pair of cameras
    stereo_params = []
    for i in range(3):
        for j in range(i + 1, 4):
            retval, camera_matrices[i], dist_coeffs[i], camera_matrices[j], dist_coeffs[j], R, T, E, F = cv2.stereoCalibrate(
                obj_points, img_points[i], img_points[j], camera_matrices[i], dist_coeffs[i], camera_matrices[j], dist_coeffs[j], grays[0].shape[::-1],
                criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
            stereo_params.append((i, j, R.astype(np.float32).tolist(), T.astype(np.float32).tolist()))

            print(f"Rotation matrix between camera {i+1} and camera {j+1}:\n", R)
            print(f"Translation vector between camera {i+1} and camera {j+1}:\n", T)

    # Save calibration results
    np.savez('multi_camera_calibration.npz', camera_matrices=camera_matrices, dist_coeffs=dist_coeffs)

    # Convert numpy arrays to lists for JSON serialization
    serializable_stereo_params = [{'camera_pair': (i, j), 'R': R, 'T': T} for (i, j, R, T) in stereo_params]

    with open('stereo_params.json', 'w') as f:
        json.dump(serializable_stereo_params, f, indent=4)

# Example usage
calibrate_multiple_cameras('cameras_calibration_images', pattern_size=(8, 6))
