import cv2

# Load the combined image
combined_image = cv2.imread('test2.jpg')

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

# Save the individual images
cv2.imwrite('image9.jpg', image1)
cv2.imwrite('image10.jpg', image2)
cv2.imwrite('image11.jpg', image3)
cv2.imwrite('image12.jpg', image4)

print("Images have been separated and saved successfully.")
