import cv2
import numpy as np


def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise Exception(f"Cannot open image: {image_path}")

    # Resize for better OCR
    scale = 2.5
    image = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # Convert to Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove Noise
    gray = cv2.fastNlMeansDenoising(gray)

    # CLAHE improves document contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # Sharpen
    kernel = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1]
    ])

    sharp = cv2.filter2D(gray, -1, kernel)

    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    return thresh