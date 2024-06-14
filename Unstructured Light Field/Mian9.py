import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_images(image_paths):
    images = [cv2.imread(image_path) for image_path in image_paths]
    return images

def align_images(images):
    aligned_images = []
    ref_img = images[0]
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    aligned_images.append(ref_img)

    for img in images[1:]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        (cc, warp_matrix) = cv2.findTransformECC(ref_gray, gray, warp_matrix, cv2.MOTION_HOMOGRAPHY)
        aligned_img = cv2.warpPerspective(img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        aligned_images.append(aligned_img)
    
    # Debug: Save aligned images
    for idx, img in enumerate(aligned_images):
        cv2.imwrite(f'aligned_image_{idx}.jpg', img)

    return aligned_images

def compute_focus_measure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus_measure = cv2.GaussianBlur(laplacian**2, (3, 3), 0)
    return focus_measure

def create_depth_map(images):
    depth_map = np.zeros(images[0].shape[:2], dtype=np.float32)
    focus_measures = np.zeros(images[0].shape[:2], dtype=np.float32)
    
    for i, img in enumerate(images):
        focus_measure = compute_focus_measure(img)
        mask = focus_measure > focus_measures
        depth_map[mask] = i
        focus_measures[mask] = focus_measure[mask]
    
    # Debug: Save focus measures and depth map
    plt.imshow(focus_measures, cmap='gray')
    plt.title('Focus Measures') 
    plt.savefig('focus_measures.jpg')
    plt.show()

    plt.imshow(depth_map, cmap='jet')
    plt.title('Depth Map')
    plt.savefig('depth_map.jpg')
    plt.show()
    
    return depth_map

def create_all_in_focus_image(images, depth_map):
    all_in_focus = np.zeros_like(images[0], dtype=np.float32)
    for i, img in enumerate(images):
        mask = (depth_map == i)
        all_in_focus[mask] = img[mask]
    all_in_focus = np.clip(all_in_focus, 0, 255).astype(np.uint8)
    return all_in_focus

def refocus_image(images, depth_map, focus_depth):
    refocused_img = np.zeros_like(images[0], dtype=np.float32)
    count = np.zeros_like(images[0], dtype=np.float32)
    
    for i, img in enumerate(images):
        mask = np.abs(depth_map - focus_depth) < 0.1  # Select pixels close to focus depth
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        refocused_img += img.astype(np.float64) * mask_3d
        count += mask_3d
    
    # Avoid division by zero
    count[count == 0] = 1
    refocused_img /= count
    
    # Debug: Check for NaN values
    if np.isnan(refocused_img).any():
        print("NaN values found in refocused image")

    refocused_img = np.clip(refocused_img, 0, 255).astype(np.uint8)
    return refocused_img

def process_focus_stacking(image_paths, focus_depth):
    images = load_images(image_paths)
    aligned_images = align_images(images)
    
    depth_map = create_depth_map(aligned_images)
    all_in_focus_img = create_all_in_focus_image(aligned_images, depth_map)
    refocused_img = refocus_image(aligned_images, depth_map, focus_depth)
    
    plt.imshow(cv2.cvtColor(all_in_focus_img, cv2.COLOR_BGR2RGB))
    plt.title("All-in-Focus Image")
    plt.savefig('all_in_focus.jpg')
    plt.show()
    
    plt.imshow(cv2.cvtColor(refocused_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Refocused Image at Depth {focus_depth}")
    plt.savefig(f'refocused_at_depth_{focus_depth}.jpg')
    plt.show()
    
    return all_in_focus_img, refocused_img

image_paths = [
'Images/Camera1/image_1.jpg',
'Images/Camera2/image_1.jpg',
'Images/Camera3/image_1.jpg',
'Images/Camera4/image_1.jpg',
'Images/Camera1/image_2.jpg',
'Images/Camera2/image_2.jpg',
'Images/Camera3/image_2.jpg',
'Images/Camera4/image_2.jpg',
'Images/Camera1/image_3.jpg',
'Images/Camera2/image_3.jpg',
'Images/Camera3/image_3.jpg',
'Images/Camera4/image_3.jpg',
'Images/Camera1/image_4.jpg',
'Images/Camera2/image_4.jpg',
'Images/Camera3/image_4.jpg',
'Images/Camera4/image_4.jpg'
]
focus_depth = 1
all_in_focus_img, refocused_img = process_focus_stacking(image_paths, focus_depth)
