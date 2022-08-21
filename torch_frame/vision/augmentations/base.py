import cv2
import random
import numpy as np


__all__ = [
    "mixup"
]


def mixup(img1: np.ndarray, img2: np.ndarray,
                mixup_scale: float = 0.1):
    def translate(image, h, w, alpha):
        y_offset = random.randint(0, dst_img.shape[0] - h)
        x_offset = random.randint(0, dst_img.shape[1] - w)
        dst_img[y_offset: h+y_offset, x_offset: w+x_offset] = image * alpha

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    dst_img = np.zeros((max(h1, h2), max(w1, w2), 3), dtype=np.float)
    alpha = random.uniform(0.5 - mixup_scale, 0.5 + mixup_scale)
    img1 = img1.astype(np.float32)
    translate(img1, h1, w1, alpha)
    translate(img1, h2, w2, 1 - alpha)
    return dst_img.astype(np.uint8)