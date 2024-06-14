import cv2
import numpy as np



def compute_focus_measures(grays):
    focus_measures = [cv2.Laplacian(gray, cv2.CV_64F).var() for gray in grays]
    return focus_measures

def compute_depth_map(focus_measures, shape):
    focus_stack = np.stack([np.full(shape, fm, dtype=np.float32) for fm in focus_measures], axis=0)
    depth_map = np.argmax(focus_stack, axis=0)
    return depth_map

def create_all_in_focus(images, depth_map):
    height, width = images[0].shape[:2]
    all_in_focus = np.zeros_like(images[0])
    for h in range(height):
        for w in range(width):
            idx = depth_map[h, w]
            all_in_focus[h, w] = images[idx][h, w]
    return all_in_focus

def calculate_image_distance(f, do):
    # Gaussian lens formula to find image distance di (both f and do in meters)
    di = 1 / ((1 / f) - (1 / do))
    return di

def scale_images(images, object_distances, focal_length):
    focal_length_m = focal_length / 1000  # Convert mm to meters
    object_distances_m = [d / 1000 for d in object_distances]  # Convert mm to meters
    image_distances = [calculate_image_distance(focal_length_m, do) for do in object_distances_m]
    base_image_distance = image_distances[0]  # Assuming the first image is the base
    scaled_images = []
    for img, di in zip(images, image_distances):
        scale_factor = base_image_distance / di
        resized_image = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        scaled_images.append(resized_image)
    return scaled_images

# Example usage
def load_images(filepaths):
    images = []
    grays = []
    for file in filepaths:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
            grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        else:
            print(f"Failed to load image: {file}")
    return images, grays
path='BonusImages/Image-'
image_paths = []
for i in range(5):
    image_path=path+str(i+1)+'.jpg'
    image_paths.append(image_path)
object_distances = [300,298,296,298,298]  
focal_length = 40  # Focal length in mm

images, grays = load_images(image_paths)
if images and grays:
    scaled_images = scale_images(images, object_distances, focal_length)
    focus_measures = compute_focus_measures([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in scaled_images])
    print("Focus measures:", focus_measures)
    depth_map = compute_depth_map(focus_measures, scaled_images[0].shape[:2])
    print("Unique values in depth map:", np.unique(depth_map))
    
    # Apply color map for better visualization
    depth_map_visual = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=(255.0/depth_map.max())), cv2.COLORMAP_JET)
    cv2.imwrite("all_in_focus_image_depth_visual.png", depth_map_visual)
    all_in_focus_image = create_all_in_focus(scaled_images, depth_map)
    cv2.imwrite("all_in_focus_image.png", all_in_focus_image)
    # Normalize and apply colormap
    depth_map_normalized = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_map_visual = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    cv2.imwrite("all_in_focus_image_depth_visual_new.png", depth_map_visual)
else:
    print("Some images could not be loaded properly.")