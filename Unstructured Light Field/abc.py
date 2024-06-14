import matplotlib.pyplot as plt
import cv2

# Load the image
image_path = "testImage.png"
image = cv2.imread(image_path)

# Convert the image from BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image_rgb)
plt.title('Refocused Image at Depth 1')
plt.axis('off')  # Hide the axis
plt.show()
