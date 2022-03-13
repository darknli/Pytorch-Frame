from __future__ import annotations
import numpy as np
import torch
import torchvision
from typing import Optional, List


__all__ = [
    "filter_box",
    "det_postprocess",
    "bboxes_iou",
    "xyxy2cxcywh",
    "box_candidates"
]


def filter_box(boxes: np.ndarray | torch.Tensor,
               min_scale: float = None,
               max_scale: float = None) -> np.ndarray | torch.Tensor:
    """
    目标框boxes根据尺寸做过滤
    Parameters
    ----------
    boxes : np.ndarray or torch.Tensor
        待过滤的目标框
    min_scale : float
    max_scale : float

    Returns
    -------
    boxes : np.ndarray or torch.Tensor
    """
    if min_scale is None:
        min_scale = 0.0
    if max_scale is None:
        max_scale = float('inf')
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return boxes[keep]


def det_postprocess(prediction: torch.Tensor,
                    num_classes: int,
                    conf_thre: float = 0.7,
                    nms_thre: float = 0.45,
                    class_agnostic: bool = False) -> List[Optional[torch.Tensor]]:
    """
    目标检测的后处理, 给定候选目标框, 经过置信度和mns双重筛选, 得到最终目标框
    Parameters
    ----------
    prediction : torch.Tensor
        检测模型给出的候选目标框, 采取了YOLO中boxes的形式: shape=(batch_size, num_boxes, 4 + 1 + num_classes)
        boxes的格式是x1,y1,x2,y2,confidence,class_score1, class_score2,...,class_scoreN
    num_classes : int
    conf_thre : float, default 0.7
        置信度阈值, 目标框的置信度分数=confidence*class_score, 过滤掉所有置信度分数小于conf_thre的boxes
    nms_thre : float, default 0.45
        mns的阈值
    class_agnostic: bool, default False

    Returns
    -------
    output : List[Optional[torch.Tensor]]
        做完置信度和mns过滤的boxes
    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a: np.ndarray | torch.Tensor, bboxes_b: np.ndarray | torch.Tensor, xyxy: bool = True):
    """
    Parameters
    ----------
    bboxes_a : np.ndarray | torch.Tensor, shape=(batch_size, 4)
    bboxes_b : np.ndarray | torch.Tensor, shape=(batch_size, 4)
    xyxy : bool, default True
        输入格式是[x1,y1,x2,y2]还是[cx,cy,w,h]
    Returns
    -------
    iou : np.ndarray | torch.Tensor, shape=(batch_size, )
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    iou = area_i / (area_a[:, None] + area_b - area_i)
    return iou


def xyxy2cxcywh(bboxes: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    输入boxes坐标, 格式从[[x1, y1, x2, y2]] -> [[cx, cy, w, h]], shape=[None, 4]

    Parameters
    ----------
    bboxes : np.ndarray or torch.Tensor
        目标框坐标数组, shape=(None, 4), 格式是[[x1, y1, x2, y2]]
    Returns
    -------
    bboxes : np.ndarray or torch.Tensor
        目标框坐标数组, shape=(None, 4), 格式是[[cx, cy, w, h]]
    """
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def box_candidates(box1: np.ndarray, box2: np.ndarray, wh_thr: int = 2, ar_thr: int = 20,
                   area_thr: float = 0.2):
    """
    对box2做过滤，对于一些极端情况做删除

    Parameters
    ----------
    box1 : np.ndarray. shape=(None, 4), 格式是[[x1, y1, x2, y2]].
    box2 : np.ndarray. shape=(None, 4), 格式是[[x1, y1, x2, y2]].
    wh_thr : int, default 2. 宽高的最小阈值, 小于该值会被过滤.
    ar_thr : int, default 20. 最大宽高比阈值, 大于该值会被过滤.
    area_thr: float, default 0.2. 与box1的面积比例，小于该值会被过滤

    Returns
    -------
    mask: np.ndarray. shape=(None, ), box2的mask.
    """
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr)
    )