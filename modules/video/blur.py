import cv2
import numpy as np


def blur_region(frame, bbox, blur_strength=51):
    """
    Applies Gaussian blur to a bounding box region.

    Args:
        frame (np.ndarray): Original frame
        bbox (tuple): (x1, y1, x2, y2)
        blur_strength (int): Kernel size (must be odd)

    Returns:
        np.ndarray: Frame with blurred region
    """

    x1, y1, x2, y2 = bbox

    # Ensure valid boundaries
    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2]

    # Ensure blur kernel is odd
    if blur_strength % 2 == 0:
        blur_strength += 1

    blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)

    frame[y1:y2, x1:x2] = blurred

    return frame