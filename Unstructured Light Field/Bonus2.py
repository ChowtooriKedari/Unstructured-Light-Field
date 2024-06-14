import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_tenengrad(gray, ksize=3):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    fm = gx**2 + gy**2  # Compute focus measure for each pixel
    return fm

def load_images(filepaths):
    images = []
    grays = []
    for file in filepaths:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grays.append(gray)
        else:
            print(f"Failed to load image: {file}")
    return images, grays

def create_depth_map(grays):
    focus_stack = np.array([compute_tenengrad(gray) for gray in grays])
    depth_map = np.argmax(focus_stack, axis=0)
    plt.imshow(depth_map, cmap='gray')
    plt.title("Depth Map")
    plt.colorbar()
    plt.show()
    return depth_map

def create_all_in_focus_image(images, depth_map):
    all_in_focus = np.zeros_like(images[0])
    for i in range(len(images)):
        mask = (depth_map == i)
        all_in_focus[mask] = images[i][mask]
    plt.imshow(cv2.cvtColor(all_in_focus, cv2.COLOR_BGR2RGB))
    plt.title("All-in-Focus Image")
    plt.show()

path = 'BonusImages/Image-'
image_paths = [f"{path}{i+1}.jpg" for i in range(4)]

images, grays = load_images(image_paths)
if images and grays:
    depth_map = create_depth_map(grays)
    create_all_in_focus_image(images, depth_map)
else:
    print("Some images could not be loaded properly.")
