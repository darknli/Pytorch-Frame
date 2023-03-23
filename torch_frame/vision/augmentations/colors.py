from random import randint, random, uniform
import numpy as np
import cv2

__all__ = [
    "RandomBrightness",
    "RandomGammaCorrection",
    "RandomHueSaturation",
    "RandomContrast",
]


class RandomBrightness:
    def __init__(self, low=-10, high=10, p=0.2):
        if low < -80 or high > 80:
            raise ValueError("亮度low不可以小于-80, high不可以大于80")
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, image):
        if random() < self.p:
            return image
        image = image.astype(np.float32)
        brightness = randint(self.low, self.high)
        image[:, :, :] += brightness
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image


class RandomGammaCorrection:
    def __init__(self, lower=0.666, upper=1.5, p=0.5):
        self.lower = lower
        self.upper = upper
        self.p = p

    def __call__(self, image):
        if random() < self.p:
            return image
        cur_gamma = uniform(self.lower, self.upper)
        max_channels = (255, 255, 255)
        image = np.power(image / max_channels, cur_gamma)
        image = (image * 255).astype(np.uint8)
        return image


class RandomHueSaturation:
    def __init__(self, hue_lower=-12, hue_upper=12, sat_lower=0.6, sat_upper=1.5, p=0.5):
        self.hue_lower = hue_lower
        self.hue_upper = hue_upper
        self.sat_lower = sat_lower
        self.sat_upper = sat_upper
        self.p = p

    def __call__(self, image):
        if random() < self.p:
            return image
        random_h = uniform(self.hue_lower, self.hue_upper)
        random_s = uniform(self.sat_lower, self.sat_upper)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = image[:, :, 0]
        h += random_h
        h[h > 180.0] -= 180.0
        h[h < 0.0] += 180.0

        s = image[:, :, 1]
        s *= random_s
        # 一定要先转uint8
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image


class RandomContrast:
    def __init__(self, low=0.9, high=1.1, p=0.2):
        if low < 0.5 or high > 2:
            raise ValueError("对比度low不可以小于0.5, high不可以大于2.0")
        if low >= high:
            raise ValueError("对比度low不可以大于或等于high")
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, image):
        if random() < self.p:
            return image
        image = image.astype(np.float32)
        contrast = uniform(self.low, self.high)
        image[:, :, :] *= contrast
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image