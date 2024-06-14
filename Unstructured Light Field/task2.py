import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from PIL import Image

def load_and_convert_images(image_paths):
    # Load images and convert to numpy arrays
    images = [np.array(Image.open(path)) for path in image_paths]
    return np.stack(images)

def compute_afi(images):
    # Compute AFI by averaging over aperture sizes for each focal setting
    return np.mean(images, axis=0)

def variance_of_afi(focal_index, afi, s, t):
    # Compute the variance of the AFI at pixel (s, t) across different focal settings
    return np.var(afi[focal_index, s, t])

def estimate_depth(afi, s, t):
    # Estimate depth by minimizing the variance of AFI across different focal settings
    result = minimize_scalar(variance_of_afi, args=(afi, s, t), bounds=(0, afi.shape[0]-1), method='bounded')
    return result.x

# Define image paths
image_paths = ['hair-a012-f41-gamma20.png', 'hair-a160-f41-gamma20.png']

# Load and convert images
images = load_and_convert_images(image_paths)

# Assuming these images are all at different focal settings but the same aperture size
# Reshape to match expected dimensions: [num_focal_settings, num_aperture_sizes, image_height, image_width]
num_focal_settings = 2
num_aperture_sizes = 1  # Since all images have the same aperture size in this example
image_height, image_width = images[0].shape[0], images[0].shape[1]
#images = images.reshape((num_focal_settings, num_aperture_sizes, image_height, image_width))

# Compute the AFI
afi = compute_afi(images)
print(afi)
# Estimate depth for a specific pixel
s, t = 50, 50  # Pixel coordinates
estimated_depth_index = estimate_depth(afi, s, t)
print(f"Estimated depth index at pixel ({s}, {t}): {estimated_depth_index}")

# Display depth map
depth_map = np.zeros((image_height, image_width))
for i in range(image_height):
    for j in range(image_width):
        depth_map[i, j] = estimate_depth(afi, i, j)

plt.imshow(depth_map, cmap='hot')
plt.colorbar()
plt.title('Estimated Depth Map')
plt.show()
