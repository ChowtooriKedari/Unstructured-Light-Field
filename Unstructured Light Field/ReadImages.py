import cv2
import numpy as np

def load_images(camera_indices):
    images = []
    for index in camera_indices:
        img = cv2.imread(f'image_{index}.jpg')
        images.append(img)
    return images

def create_light_field(images):
    # Assuming all images are of the same size
    height, width, _ = images[0].shape
    light_field = np.zeros((height, width, len(images), 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        light_field[:, :, i, :] = img
    
    return light_field

if __name__ == "__main__":
    camera_indices = [0, 1, 2, 3]  # Adjust based on your quad camera adapter
    images = load_images(camera_indices)
    light_field = create_light_field(images)
    
    # Save or process the light field further as needed
    np.save('light_field.npy', light_field)
    print('Light field created and saved.')
