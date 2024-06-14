import os
import json
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.interpolate import interp2d

def capture_images_indefinitely(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metadata = {}
    batch = 1
    
    while True:
        batch_folder = os.path.join(output_folder, f'batch_{batch}')
        if not os.path.exists(batch_folder):
            os.makedirs(batch_folder)

        camera_positions = []
        for cam_idx in range(4):  # Adjust the range for the number of cameras
            try:
                x = float(input(f"Enter the X coordinate for camera {cam_idx+1}: "))
                y = float(input(f"Enter the Y coordinate for camera {cam_idx+1}: "))
                z = float(input(f"Enter the Z coordinate for camera {cam_idx+1}: "))
                camera_positions.append([x, y, z])
            except ValueError:
                print("Invalid input. Please enter numeric values for coordinates.")
                continue

        metadata[f'batch_{batch}'] = {"positions": camera_positions, "images": {}}

        image_path = os.path.join(batch_folder, f'batch_{batch}_image.jpg')
        command = f'libcamera-still -t 5000 -n -o {image_path}'
        subprocess.run(command, shell=True)

        combined_image = cv2.imread(image_path)
        plt.imshow(combined_image)
        plt.show()
        
        height, width, _ = combined_image.shape
        individual_height = height // 2
        individual_width = width // 2

        image1 = combined_image[0:individual_height, 0:individual_width]
        image2 = combined_image[0:individual_height, individual_width:width]
        image3 = combined_image[individual_height:height, 0:individual_width]
        image4 = combined_image[individual_height:height, individual_width:width]
        plt.imshow(image1)
        plt.show()

        cv2.imwrite(f'Images/Camera1/image_{batch}.jpg', image1)
        cv2.imwrite(f'Images/Camera2/image_{batch}.jpg', image2)
        cv2.imwrite(f'Images/Camera3/image_{batch}.jpg', image3)
        cv2.imwrite(f'Images/Camera4/image_{batch}.jpg', image4)

        for cam_idx in range(4):
            path = f'Images/Camera{cam_idx+1}/image_{batch}.jpg'
            if os.path.exists(path):
                metadata[f'batch_{batch}']["images"][f'camera_{cam_idx+1}_image_{batch}'] = path
                print(f"Captured image for camera {cam_idx+1} at {path}")
            else:
                print(f"Failed to capture image from camera {cam_idx+1}")

        print(f"Captured batch {batch}")

        with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        user_input = input("Press Enter to capture next batch, or type 'stop' to end: ")
        if user_input.lower() == 'stop':
            print("Stopping the capture process.")
            break

        batch += 1

# Example usage
output_folder = 'calibration_images'
capture_images_indefinitely(output_folder)
