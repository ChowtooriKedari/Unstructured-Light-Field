import os
import json
import subprocess
import cv2

import numpy as np
import matplotlib.pyplot as plt
from Helper import lRGB2XYZ  # Assuming loadVideo is not needed anymore
from scipy.signal import fftconvolve
from scipy.interpolate import interp2d
import cv2
import glob

def capture_images_indefinitely(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(1,5):             
        if not os.path.exists(f'Images/Camera{i}'):
            os.makedirs(f'Images/Camera{i}')
    metadata = {}
    batch = 1
    
    while True:
        batch_folder = os.path.join(output_folder, f'batch_{batch}')
        if not os.path.exists(batch_folder):
            os.makedirs(batch_folder)

        if batch == 1:
            position = [0, 0, 0]
        else:
            try:
                x = float(input("Enter the X coordinate for the new position: "))
                y = float(input("Enter the Y coordinate for the new position: "))
                z = float(input("Enter the Z coordinate for the new position: "))
                position = [x, y, z]
            except ValueError:
                print("Invalid input. Please enter numeric values for coordinates.")
                continue

        metadata[f'batch_{batch}'] = {"position": position, "images": {}}

        image_path = os.path.join(batch_folder, f'batch_{batch}_image.jpg')
        command = f'libcamera-still -t  5000 -n -o {image_path}'
        subprocess.run(command, shell=True)

                # Load the combined image
        combined_image = cv2.imread(image_path)
        plt.imshow(combined_image)
        plt.show()
        # Get the dimensions of the combined image
        height, width, _ = combined_image.shape

        # Assuming a 2x2 grid for 4 cameras
        individual_height = height // 2
        individual_width = width // 2

        # Split the combined image into four individual images
        image1 = combined_image[0:individual_height, 0:individual_width]
        image2 = combined_image[0:individual_height, individual_width:width]
        image3 = combined_image[individual_height:height, 0:individual_width]
        image4 = combined_image[individual_height:height, individual_width:width]
        plt.imshow(image1)
        plt.show()

        # Save the individual images
        cv2.imwrite(f'Images/Camera1/image_{batch}.jpg', image1)
        cv2.imwrite(f'Images/Camera2/image_{batch}.jpg', image2)
        cv2.imwrite(f'Images/Camera3/image_{batch}.jpg', image3)
        cv2.imwrite(f'Images/Camera4/image_{batch}.jpg', image4)


        for cam_idx in range(4):  # Adjust the range as needed for the number of cameras           
            path='Images/Camera'+str(cam_idx+1)+'/image_'+str(batch)+'.jpg'
            if os.path.exists(path):
                    metadata[f'batch_{batch}']["images"][f'camera_{cam_idx+1}_image_{batch}'] = path
                    print(f"Captured image for camera {cam_idx+1} at {image_path}")
            else:
                    print(f"Failed to capture image from camera {cam_idx+1}")

        print(f"Captured batch {batch}")

            # Save the metadata after each batch
        with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)

        # Ask user if they want to continue or stop
        user_input = input("Press Enter to capture next batch, or type 'stop' to end: ")
        if user_input.lower() == 'stop':
            print("Stopping the capture process.")
            break

        # Increment the batch number
        batch += 1

# Example usage
output_folder = 'calibration_images'
capture_images_indefinitely(output_folder)
