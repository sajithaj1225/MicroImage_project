
#This is to split the images into 4 overlapping parts


import cv2
import numpy as np
import os

# Loading the original image
img = cv2.imread("images/microscopic_img.jpg")

# Check if the image is loaded properly
if img is None:
    print("Error: Could not load the image. Check the file path.")
    exit()

# Get the width and height
height, width, _ = img.shape

# Define the overlap percentage
overlap = int(width * 0.1)  # Reduced overlap to 10%

# Split the image into 4 overlapping parts
img1 = img[:, :width//4 + overlap]  # First quarter + overlap
img2 = img[:, width//4 - overlap: width//2 + overlap]  # Second quarter + overlap
img3 = img[:, width//2 - overlap: (3*width)//4 + overlap]  # Third quarter + overlap
img4 = img[:, (3*width)//4 - overlap:]  # Fourth quarter + overlap

# Save the split images
cv2.imwrite("images/image1.jpg", img1)
cv2.imwrite("images/image2.jpg", img2)
cv2.imwrite("images/image3.jpg", img3)
cv2.imwrite("images/image4.jpg", img4)

print("✅ Image successfully split into 4 overlapping parts.")
