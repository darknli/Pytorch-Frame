from typing import Union
import numpy as np
import cv2
import random
import math
from ..tools.object_detection import box_candidates

__all__ = [
    "random_perspective",
    "mixup_boxes",
    "mosaic"
]


def random_perspective(img: np.ndarray, targets: Union[np.ndarray, list, tuple], angle: int = 10, translate: int = 0.1,
                       scale: tuple = (0.7, 1.5), shear: float = 2.0, border: tuple = (0, 0),
                       fill_value: tuple = (128, 128, 128)):
    """
    对img做随机仿射变换，并且对应的boxes也做相应变化

    Parameters
    ----------
    img : np.ndarray. 原始图像.
    targets : np.ndarray, list or tuple. 目标box列表, 格式为[[x1, y1, x2, y2], ..., [x1, y1, x2, y2]].
    angle : float or int. 旋转图像的最大度数（含正负）.
    translate : int, default 0.1. 随机沿着宽高平移的百分比, 比如0.1代表分别向x和y方向平移宽和高的最大0.1距离.
    scale : tuple, default (0.7, 1.5). 缩放的范围, 第一个值代表最小界限，第二个值代表最大界限.
    shear : float, default 2. 仿射变换的不规则形变的范围, 如果是0，那么就不做仿射变换.
    border : tuple, default (0, 0). 向外padding的个数， 第一个参数是高height，第二个是宽width.
    fill_value: tuple, default (128, 128, 128). padding填充的颜色

    Returns
    -------
    img : np.ndarray. 做过仿射变换的图片
    targets : np.ndarray. 经过仿射变换后做过过滤的boxes
    """
    if isinstance(targets, tuple) or isinstance(targets, tuple):
        targets = np.array(targets, dtype=np.float)

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    padding_image = np.full((height, width, 3), fill_value, dtype=np.uint8)
    padding_image[border[0]:-border[0], border[1]:-border[1]] = img
    img = padding_image

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-angle, angle)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    perspective = abs(shear) < 1e-3

    # Translation
    T = np.eye(3)
    T[0, 2] = (
            random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
            random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT, @ means matrix multiplication

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=fill_value)
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=fill_value)

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

    return img, targets


def mixup_boxes(img1: np.ndarray, boxes1: np.ndarray, img2: np.ndarray, boxes2: np.ndarray,
                mixup_scale: float = 0.1, wh_thr: int = 0, ar_thr: int = 100, area_thr: float = 0):
    """
    mixup的目标检测版

    Parameters
    ----------
    img1 : np.ndarray. 图片1
    boxes1 : np.ndarray. 图片1对应的boxes
    img2 : np.ndarray. 图片2
    boxes2 : np.ndarray. 图片2对应的boxes
    mixup_scale: float, default 0.1. 做alpha融合的alpha波动概率，比如0.1的话实际alpha值会在0.4~0.6均匀采样
    wh_thr : int, default 0. boxes宽和高的最小阈值，低于它的box会被删除
    ar_thr ： int, default 100. 宽高比，低于它的box会被删除
    area_thr : float, default 0. 旧vs新box面积比例，低于它的box被删除

    Returns
    -------
    dst_img : np.ndarray
    dst_boxes : np.ndarray
    """
    def translate(image, boxes, h, w, alpha):
        y_offset = random.randint(0, dst_img.shape[0] - h)
        x_offset = random.randint(0, dst_img.shape[1] - w)
        dst_img[y_offset: h + y_offset, x_offset: w + x_offset] += image * alpha
        if len(boxes) > 0:
            new_boxes = boxes.copy()
            new_boxes[:, 1::2] = np.clip(new_boxes[:, 1::2] + x_offset, 0, dst_w)
            new_boxes[:, 2::2] = np.clip(new_boxes[:, 2::2] + y_offset, 0, dst_h)
            mask = box_candidates(boxes, new_boxes[:, 1:], wh_thr, ar_thr, area_thr)
            boxes = new_boxes[mask]
        else:
            boxes = np.zeros((0, 5))
        return dst_img, boxes

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    dst_img = np.zeros((max(h1, h2), max(w1, w2), 3), dtype=np.float)
    dst_h, dst_w = dst_img.shape[:2]
    alpha = random.uniform(0.5 - mixup_scale, 0.5 + mixup_scale)
    img1 = img1.astype(np.float32)
    dst_img, dst_boxes1 = translate(img1.astype(float), boxes1, h1, w1, alpha)
    dst_img, dst_boxes2 = translate(img2.astype(float), boxes2, h2, w2, 1 - alpha)
    dst_boxes = np.vstack([dst_boxes1, dst_boxes2])
    return dst_img.astype(np.uint8), dst_boxes


def mosaic(data: list, ouput_dim: Union[list, tuple], fill_value=114):
    """
    mosaic拼接，用于数据增强，对于小目标检测有效

    Parameters
    ----------
    data : List[tuple]. list包含(image, bboxes)tuple，长度是4
    ouput_dim : Union[list, tuple]. 输出是[height, width]
    fill_value : Union[int, tuple].

    Returns
    -------
    mosaic_img : np.ndarray. 拼接后的图片
    mosaic_bboxes : Union[list, np.ndarray]. 拼接后的目标框，为空时是list类型
    """

    def get_mosaic_coordinate(mosaic_index, xc, yc, w, h):
        # index0 to top left part of image
        if mosaic_index == 0:
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coord = w - (x2 - x1), h - (y2 - y1), w, h
        # index1 to top right part of image
        elif mosaic_index == 1:
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, ouput_w * 2), yc
            small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        # index2 to bottom left part of image
        elif mosaic_index == 2:
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(ouput_h * 2, yc + h)
            small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        # index2 to bottom right part of image
        elif mosaic_index == 3:
            x1, y1, x2, y2 = xc, yc, min(xc + w, ouput_w * 2), min(ouput_h * 2, yc + h)  # noqa
            small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
        return (x1, y1, x2, y2), small_coord

    assert len(data) == 4, "`data`长度应该是4"
    assert len(ouput_dim) == 2, "`data`长度应该是2"

    mosaic_bboxes = []
    ouput_h, ouput_w = ouput_dim[0], ouput_dim[1]

    # yc, xc = s, s  # mosaic center x, y
    yc = int(random.uniform(0.5 * ouput_h, 1.5 * ouput_h))
    xc = int(random.uniform(0.5 * ouput_w, 1.5 * ouput_w))
    mosaic_img = np.full((ouput_h * 2, ouput_w * 2, 3), fill_value, dtype=np.uint8)

    for i_mosaic, (img, bboxes) in enumerate(data):
        h0, w0 = img.shape[:2]  # orig hw
        scale = min(1. * ouput_h / h0, 1. * ouput_w / w0)
        img = cv2.resize(
            img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
        )
        h, w = img.shape[:2]

        # suffix l means large image, while s means small image in mosaic aug.
        (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(i_mosaic, xc, yc, w, h)

        mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
        padw, padh = l_x1 - s_x1, l_y1 - s_y1

        # Normalized xywh to pixel xyxy format
        if bboxes.size > 0:
            scale_bboxes = bboxes.copy()
            scale_bboxes[:, 1] = scale * bboxes[:, 1] + padw
            scale_bboxes[:, 2] = scale * bboxes[:, 2] + padh
            scale_bboxes[:, 3] = scale * bboxes[:, 3] + padw
            scale_bboxes[:, 4] = scale * bboxes[:, 4] + padh
        else:
            scale_bboxes = np.zeros((0, 5))
        mosaic_bboxes.append(scale_bboxes)

    if len(mosaic_bboxes) > 0:
        mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
        np.clip(mosaic_bboxes[:, 0], 0, 2 * ouput_w, out=mosaic_bboxes[:, 0])
        np.clip(mosaic_bboxes[:, 1], 0, 2 * ouput_h, out=mosaic_bboxes[:, 1])
        np.clip(mosaic_bboxes[:, 2], 0, 2 * ouput_w, out=mosaic_bboxes[:, 2])
        np.clip(mosaic_bboxes[:, 3], 0, 2 * ouput_h, out=mosaic_bboxes[:, 3])

    return mosaic_img, mosaic_bboxes
