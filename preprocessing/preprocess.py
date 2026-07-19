"""
MRI Preprocessing Pipeline
Matches base paper (Yadav et al., 2025):
1. Load and convert to grayscale
2. Resize to 224x224
3. Object-centric ROI extraction (contour detection)
4. Gamma correction
5. CLAHE contrast enhancement
6. Normalize [0,1] → 3-channel for EfficientNet
"""

import cv2
import numpy as np
from PIL import Image
import io


def preprocess_mri(image_path: str = None, image_bytes: bytes = None, target_size=(224, 224)) -> np.ndarray:
    """
    Full preprocessing pipeline.
    Accepts either a file path or raw bytes.
    Returns: float32 numpy array of shape (224, 224, 3), values in [0, 1].
    """
    # Step 1: Load image
    if image_bytes is not None:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif image_path is not None:
        img = cv2.imread(image_path)
    else:
        raise ValueError("Either image_path or image_bytes must be provided.")

    if img is None:
        raise ValueError("Failed to load image. Check the file path or bytes.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize
    img_resized = cv2.resize(gray, target_size)

    # Step 3: ROI extraction (object-centric)
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    _, thresh = cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        extLeft  = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop   = tuple(c[c[:, :, 1].argmin()][0])
        extBot   = tuple(c[c[:, :, 1].argmax()][0])
        img_roi  = img_resized[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        if img_roi.size > 0:
            img_resized = cv2.resize(img_roi, target_size)

    # Step 4: Gamma correction
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    img_gamma = cv2.LUT(img_resized, table)

    # Step 5: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gamma)

    # Step 6: Normalize
    img_norm = img_clahe / 255.0

    # Convert to 3-channel for EfficientNet
    img_rgb = np.stack([img_norm] * 3, axis=-1)

    return img_rgb.astype(np.float32)
