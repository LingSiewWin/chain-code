import cv2
import numpy as np

# Load image as binary
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Function to get 8-direction chain code
def get_chain_code(contour):
    chain_code = []
    for i in range(1, len(contour)):
        dy = contour[i][0][1] - contour[i-1][0][1]
        dx = contour[i][0][0] - contour[i-1][0][0]
        direction = {
            (1, 0): 0,
            (1, -1): 1,
            (0, -1): 2,
            (-1, -1): 3,
            (-1, 0): 4,
            (-1, 1): 5,
            (0, 1): 6,
            (1, 1): 7
        }.get((dx, dy))
        if direction is not None:
            chain_code.append(direction)
    return chain_code

# Use the first contour
code = get_chain_code(contours[0])
print("Chain code:", code)
