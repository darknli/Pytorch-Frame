from __future__ import annotations

from typing import Optional, List

import numpy as np
import torch
import torchvision

__all__ = [
    "filter_box",
    "det_postprocess",
    "bboxes_iou",
    "xyxy2cxcywh",
    "box_candidates",
    "eval_map"
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
                    class_agnostic: bool = False,
                    merge_score: bool = False,
                    ret_np: bool = False
                    ) -> List[Optional[torch.Tensor]]:
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
    merge_score: bool, default False. 如果是True则会把obj_score和cls_score合并,返回的output是
        [x1,y1, x2, y2, obj_score*cls_score, cls]
    ret_np: bool, default False. 是否返回np.ndarray类型结果

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
        if merge_score:
            detections = torch.cat(
                [
                    detections[:, :4],
                    torch.unsqueeze(detections[:, 4] * detections[:, 5], 1),
                    torch.unsqueeze(detections[:, 6], 1)
                ],
                dim=1
            )
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
        if ret_np:
            output[i] = output[i].cpu().numpy()
    return output


def bboxes_iou_torch(
        bboxes_a: torch.Tensor,
        bboxes_b: torch.Tensor,
        xyxy: bool = True,
        mode: str = "iou",
        eps: float = 1e-6
):
    """
    Parameters
    ----------
    bboxes_a : torch.Tensor, shape=(batch_size, 4)
    bboxes_b : torch.Tensor, shape=(batch_size, 4)
    xyxy : bool, default True
        输入格式是[x1,y1,x2,y2]还是[cx,cy,w,h]
    mode : str, default `iou`. 支持在以下两种种类型中选择
        * `iou`
        * `iof`
    eps : float, default 1e-6. 为防止除以0引入的小数

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

    if mode == 'iou':
        return area_i / torch.clip(area_a[:, None] + area_b - area_i, eps)
    elif mode == 'iof':
        return area_i / torch.clip(torch.minimum(area_a[:, None], area_b), eps)
    else:
        raise ValueError("mode没有这种模式")


def bboxes_iou_numpy(
        bboxes_a: np.ndarray,
        bboxes_b: np.ndarray,
        xyxy: bool = True,
        mode: str = "iou",
        eps: float = 1e-6
):
    """
    Parameters
    ----------
    bboxes_a : np.ndarray | torch.Tensor, shape=(batch_size, 4)
    bboxes_b : np.ndarray | torch.Tensor, shape=(batch_size, 4)
    xyxy : bool, default True
        输入格式是[x1,y1,x2,y2]还是[cx,cy,w,h]
    mode : str, default `iou`. 支持在以下两种种类型中选择
        * `iou`
        * `iof`
    eps : float, default 1e-6. 为防止除以0引入的小数

    Returns
    -------
    iou : np.ndarray | torch.Tensor, shape=(batch_size, )
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = np.maximum(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = np.minimum(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = np.maximum(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = np.minimum(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = np.prod(bboxes_a[:, 2:], 1)
        area_b = np.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).astype(float).prod(axis=2)
    area_i = np.prod(br - tl, 2) * en  # * ((tl < br).all())

    if mode == 'iou':
        return area_i / np.maximum(area_a[:, None] + area_b - area_i, eps)
    elif mode == 'iof':
        return area_i / np.maximum(np.minimum(area_a[:, None], area_b), eps)
    else:
        raise ValueError("mode没有这种模式")


def bboxes_iou(
        bboxes_a: np.ndarray | torch.Tensor,
        bboxes_b: np.ndarray | torch.Tensor,
        xyxy: bool = True,
        mode: str = "iou",
        eps: float = 1e-6
):
    if isinstance(bboxes_a, np.ndarray) and isinstance(bboxes_b, np.ndarray):
        return bboxes_iou_numpy(bboxes_a, bboxes_b, xyxy, mode, eps)
    elif isinstance(bboxes_a, torch.Tensor) and isinstance(bboxes_b, torch.Tensor):
        return bboxes_iou_torch(bboxes_a, bboxes_b, xyxy, mode, eps)
    else:
        raise TypeError("`bboxes_a`和`bboxes_b`应为np.ndarray或torch.Tensor, 且相互类型一致")


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


def cal_match(pred_box, gt_box):
    count = 0
    iou = bboxes_iou(pred_box[:, 1:], gt_box[:, 1:])
    for i, pbox in enumerate(pred_box):
        for j, gbox in enumerate(gt_box.copy()):
            if iou[i][j] >= 0.9 and np.abs(pbox[1:] - gbox[1:]).max() < 5 and pbox[0] == gbox[0]:
                count += 1
                gt_box = np.concatenate([gt_box[:j], gt_box[j + 1:]])
                iou = np.concatenate([iou[:, :j], iou[:, j + 1:]], 1)
                break
    return count / len(pred_box)


def calc_ap(recall, precision, use_07_metric=True):
    # 是否使用 07 年的 11 点均值方式计算 ap
    if use_07_metric:
        ap = 0.
        for threshold in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= threshold) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= threshold])
            ap = ap + p / 11.
    else:
        # 增加哨兵，然后算出准确率包络
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 计算 pr 曲线中，召回率变化的下标
        idx = np.where(mrec[1:] != mrec[:-1])[0]

        # 计算 pr 曲线与坐标轴所围区域的面积
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return ap


def clac_voc_ap(annotation, prediction, ovthresh=0.5):
    # 统计标注框个数，这里需要将困难样本数量去掉
    positive_num = 0
    for anno in annotation:
        positive_num += len(anno.get("bboxes", []))
        anno['b_det'] = [False] * len(anno.get("bboxes", []))

    # 将检测结果格式进行转换，主要是为了方便排序
    # for item in prediction:
    image_ids, confidences, bboxes = [], [], []
    for img_id, item in enumerate(prediction):
        bbox = item.get("bboxes", [])
        score = item.get("confidence", [])
        for i in range(len(score)):
            image_ids.append(img_id)
            confidences.append(score[i])
            bboxes.append(bbox[i])
    image_ids, confidences, bboxes = np.array(image_ids), np.array(confidences), np.array(bboxes)

    # 按照置信度排序
    sorted_ind = np.argsort(-confidences)
    bboxes = bboxes[sorted_ind]
    image_ids = image_ids[sorted_ind]

    # 计算 TP 和 FP，以计算出 AP 值
    detect_num = len(image_ids)
    tp = np.zeros(detect_num)
    fp = np.zeros(detect_num)
    for d in range(detect_num):
        gt_bboxes = annotation[image_ids[d]].get("bboxes", [])

        # 如果没有 ground truth，那么所有的检测都是错误的
        if len(gt_bboxes) > 0:
            b_dets = annotation[image_ids[d]]["b_det"]
            p_bboxes = bboxes[d, :]

            overlaps = bboxes_iou(p_bboxes[np.newaxis], gt_bboxes, mode="iou")[0]
            idxmax = np.argmax(overlaps)
            ovmax = overlaps[idxmax]

            if ovmax > ovthresh:
                # gt 只允许检测出一次
                if not b_dets[idxmax]:
                    tp[d] = 1.
                    b_dets[idxmax] = True
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # 计算召回率和准确率
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / float(positive_num)
    precision = tp / np.maximum(tp + fp, 1.)
    ap = calc_ap(recall, precision)

    return recall, precision, ap


def eval_map(annotation_result, detector_result, ovthresh=0.5):
    aps = {}
    for cls in annotation_result.keys():
        recall, precision, ap = clac_voc_ap(annotation_result[cls], detector_result[cls], ovthresh=ovthresh)
        aps[cls] = ap
    return aps
