def load_images_with_positions(metadata_path, calibration_data):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    images = []
    positions = []
    for batch_key, batch_data in metadata.items():
        cam_images = []
        cam_positions = batch_data["positions"]
        for img_idx, img_path in enumerate(batch_data["images"].values()):
            img = cv2.imread(img_path)
            if img is not None:
                cam_images.append(img)
        images.append(cam_images)
        positions.append(cam_positions)
    
    # Undistort images using calibration data
    undistorted_images = []
    for cam_idx, cam_images in enumerate(images):
        undistorted_cam_images = []
        for img_idx, img in enumerate(cam_images):
            cam_matrix = np.array(calibration_data['camera_matrices'][img_idx])
            dist_coeffs = np.array(calibration_data['dist_coeffs'][img_idx])
            undistorted_img = cv2.undistort(img, cam_matrix, dist_coeffs)
            undistorted_cam_images.append(undistorted_img)
        undistorted_images.append(undistorted_cam_images)
    
    return undistorted_images, positions

def refocus_images_unstructured(images, depth_maps, positions, focal_depth):
    h, w = depth_maps[0].shape
    num_cameras = len(images)
    refocused_image = np.zeros_like(images[0][0], dtype=np.float32)
    num_images_per_camera = len(images[0])
    
    for cam_idx in range(num_cameras):
        depth_map = depth_maps[cam_idx]
        for img_idx, img in enumerate(images[cam_idx]):
            cam_pos = positions[cam_idx][img_idx]
            depth_difference = depth_map - focal_depth
            refocused_image += np.dstack((interp_unstructured(img[:, :, 0], depth_difference + cam_pos[0], depth_difference + cam_pos[1]),
                                          interp_unstructured(img[:, :, 1], depth_difference + cam_pos[0], depth_difference + cam_pos[1]),
                                          interp_unstructured(img[:, :, 2], depth_difference + cam_pos[0], depth_difference + cam_pos[1])))
    refocused_image /= (num_cameras * num_images_per_camera)
    return refocused_image.astype(np.uint8)
