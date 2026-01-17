import cv2

def blur_region(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return frame

    blurred = cv2.GaussianBlur(roi, (99, 99), 30)
    frame[y1:y2, x1:x2] = blurred
    return frame

