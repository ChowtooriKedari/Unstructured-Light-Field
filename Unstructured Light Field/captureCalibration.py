import cv2
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from Helper import lRGB2XYZ  # Assuming loadVideo is not needed anymore
from scipy.signal import fftconvolve
from scipy.interpolate import interp2d
import cv2
import glob

def capture_calibration_images(batch_folder, num_images=20, pattern_size=(8, 6)):
    if not os.path.exists(batch_folder):
            os.makedirs(batch_folder)

    metadata = {}
    batch = 1
    
    while True:
        #batch_folder = os.path.join(output_folder, f'batch_{batch}')

        if batch == 1:
            position = [0, 0, 0]  # Start at origin for the first batch
        else:
            # Get user input for the new position
            try:
                x = float(input("Enter the X coordinate for the new position: "))
                y = float(input("Enter the Y coordinate for the new position: "))
                z = float(input("Enter the Z coordinate for the new position: "))
                position = [x, y, z]
            except ValueError:
                print("Invalid input. Please enter numeric values for coordinates.")
                continue

       #metadata[f'batch_{batch}'] = {"position": position, "images": {}}

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
        cv2.imwrite(f'{batch_folder}/calibration_image_cam{batch}_Camera1.jpg', image1)
        cv2.imwrite(f'{batch_folder}/calibration_image_cam{batch}_Camera2.jpg', image2)
        cv2.imwrite(f'{batch_folder}/calibration_image_cam{batch}_Camera3.jpg', image3)
        cv2.imwrite(f'{batch_folder}/calibration_image_cam{batch}_Camera4.jpg', image4)

        # for cam_idx in range(4):  # Adjust the range as needed for the number of cameras           
        #     path='Images/Camera'+str(cam_idx+1)+'/image_'+str(batch)+'.jpg'
        #     if os.path.exists(path):
        #             metadata[f'batch_{batch}']["images"][f'camera_{cam_idx+1}_image_{batch}'] = path
        #             print(f"Captured image for camera {cam_idx+1} at {image_path}")
        #     else:
        #             print(f"Failed to capture image from camera {cam_idx+1}")

        print(f"Captured batch {batch}")

        # Ask user if they want to continue or stop
        user_input = input("Press Enter to capture next batch, or type 'stop' to end: ")
        if user_input.lower() == 'stop':
            print("Stopping the capture process.")
            break

        # Increment the batch number
        batch += 1

# Example usage:
capture_calibration_images('cameras_calibration_images', num_images=20, pattern_size=(8, 6))
