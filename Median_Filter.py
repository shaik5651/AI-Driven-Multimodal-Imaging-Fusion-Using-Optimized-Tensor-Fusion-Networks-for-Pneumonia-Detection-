import cv2
import numpy as np
from matplotlib import pyplot as plt

def artifact_processing(img_path):
    # Load image (grayscale for simplicity)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply median filter
    # ksize must be an odd number (e.g., 3, 5, 7)
    filtered_image = cv2.medianBlur(image, ksize=3)

    # Display original and filtered images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Median Filtered Image")
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    return filtered_image
