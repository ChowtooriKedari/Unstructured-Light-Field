import cv2
import numpy as np
import itertools
import glob
import matplotlib.pyplot as plt

# Function to estimate depth from multiple images using stereo vision
def estimate_depth_all(images):
    print('Estimating Depth')
    all_correspondences = []
    for image_pair in itertools.combinations(images, 2):
        correspondences = find_correspondences(*image_pair)
        all_correspondences.append(correspondences)
    depth_maps = [triangulate_depth(correspondences, images[0].shape) for correspondences in all_correspondences]
    combined_depth_map = np.mean(depth_maps, axis=0)
    return combined_depth_map

# Function to find correspondences between stereo images using SIFT and BFMatcher
def find_correspondences(image_left, image_right):
    print('Find Correspondences')
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(image_left, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(image_right, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    matched_points_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matched_points_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return matched_points_left, matched_points_right

# Function to triangulate depth from stereo correspondences
def triangulate_depth(correspondences, image_shape):
    print('Traingulate Depth')
    focal_length = 64
    baseline = 10
    matched_points_left, matched_points_right = correspondences
    depth_map = np.zeros(image_shape[:2])
    for i in range(len(matched_points_left)):
        x_l, y_l = matched_points_left[i][0]
        x_r, y_r = matched_points_right[i][0]
        disparity = x_l - x_r
        if disparity > 0:
            depth = focal_length * baseline / disparity
            depth_map[int(y_l), int(x_l)] = depth
    return depth_map

# Function to reconstruct the light field from coplanar images
def reconstruct_light_field(images):
    print('Reconstruct Light Field')
    disparity_maps = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            disparity_maps.append(find_correspondences(images[i], images[j]))
    light_field = combine_disparity_maps(disparity_maps, images[0].shape)
    return light_field

# Function to combine disparity maps to estimate light field
def combine_disparity_maps(disparity_maps, image_shape):
    print('Working on disparity maps')
    combined_disparity_map = np.zeros(image_shape[:2])
    count_map = np.zeros(image_shape[:2])
    for left_points, right_points in disparity_maps:
        for idx in range(len(left_points)):
            x_l, y_l = int(left_points[idx][0][0]), int(left_points[idx][0][1])
            x_r, y_r = int(right_points[idx][0][0]), int(right_points[idx][0][1])
            disparity = x_l - x_r
            if disparity != 0:
                combined_disparity_map[y_l, x_l] += disparity
                count_map[y_l, x_l] += 1
    combined_disparity_map[count_map > 0] /= count_map[count_map > 0]
    return combined_disparity_map

# Function to refocus an image using the depth map and light field
def refocus_image(image, depth_map, light_field, target_depth):
    print('Refocus Image')
    normalized_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    defocus_kernel = calculate_defocus_kernel(normalized_depth, target_depth)
    refocused_image = cv2.filter2D(image, -1, defocus_kernel)
    return refocused_image

# Function to calculate defocus kernel based on depth difference
def calculate_defocus_kernel(normalized_depth, target_depth):
    print('Calcuating Defocus')
    depth_difference = np.abs(normalized_depth - target_depth)
    defocus_sigma = 10 * depth_difference
    defocus_kernel = np.exp(-0.5 * (defocus_sigma ** 2))
    return defocus_kernel

# Load images

image_files = glob.glob('Images/*.jpg')  # Adjust the path to your images
images = [cv2.imread(img) for img in image_files]
for image in images:
    plt.imshow(image)
    plt.show()
# Step 1: Estimate depth field
depth_map = estimate_depth_all(images)

# Normalize depth map for visualization
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_normalized = np.uint8(depth_map_normalized)

cv2.imwrite("depth_map.png", depth_map_normalized)

# Step 2: Reconstruct unstructured light field
light_field = reconstruct_light_field(images)
cv2.imwrite("light_field.png", light_field)

# Step 3: Refocus images using depth map and light field
target_depth = 0.5
refocused_images = [refocus_image(image, depth_map, light_field, target_depth) for image in images]
for i, refocused_image in enumerate(refocused_images):
    cv2.imwrite(f"refocused_image_{i + 1}.png", refocused_image)
