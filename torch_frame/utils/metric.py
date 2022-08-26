"""
该脚本包含主流评估方法, 区别于history_buffer, 没有窗口大小的概念, 换言之不需要做平滑之类的操作, 而求所有数据的精准指标
"""
from abc import abstractmethod
from functools import partial
from typing import Dict, Optional, Callable

import numpy as np

from ..vision import det_postprocess, eval_map


class BaseMetric:
    """评估器基类, 继承它需要实现`update`, `evaluate`和`reset`方法"""

    @abstractmethod
    def update(self, *args, **kwargs):
        ...

    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def reset(self):
        ...


class ModelMetric(BaseMetric):
    """
    模型评估器将update拆分成模型推理的inference和结果收集步骤.
    继承它需要实现`inference`, `evaluate`和`reset`方法
    """

    def update(self, model, inputs):
        result = self.inference(model, inputs)
        self.collection(result)

    @abstractmethod
    def inference(self, model, inputs):
        ...

    @abstractmethod
    def collection(self, item):
        ...

    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def reset(self):
        ...


class ObjDetMAPMetric(ModelMetric):
    """
    目标检测中的AP评估器
    """

    def __init__(self,
                 class2idx: Dict[str, int],
                 is_xyxy: bool = False,
                 ovthresh: float = 0.5,
                 postprocess: Optional[Callable] = None,
                 confthre: float = 0.7,
                 nmsthre: float = 0.45,
                 ):
        """
        Parameters
        ----------
        class2idx : Dict[str, int]. 类别到索引的映射, 不包含背景类
        is_xyxy: bool, default False. gt框类型
            * False. gt目标框是[cx, cy, w, h]类型
            * True. gt目标框是[x1, y1, x2, y2]类型
        ovthresh : float, default 0.5. 计算AP时，pred和gt被认为是TP（true positive）的iou阈值
        postprocess : Optional[Callable], default None. 承接模型推理后的后处理
            * None, 则默认Yolo风格的内置函数det_postprocess
            * Callable, 返回是一个list或np.ndarray. 每一行是[x1, y1, x2, y2, score, cls]
        confthre : float, default 0.7. 如果postprocess使用默认值, 则为det_postprocess的对应参数, postprocess非空不生效!
        nmsthre : float, default 0.45. 如果postprocess使用默认值, 则为det_postprocess的对应参数, postprocess非空不生效!
        """
        self.class2idx = class2idx
        self.is_xyxy = is_xyxy
        self.ovthresh = ovthresh
        self.idx2class = {cls: idx for cls, idx in class2idx.items()}
        if postprocess is not None:
            self.postprocess = postprocess
        else:
            self.postprocess = partial(det_postprocess,
                                       num_classes=len(self.class2idx),
                                       conf_thre=confthre,
                                       nms_thre=nmsthre,
                                       class_agnostic=True,
                                       merge_score=True,
                                       ret_np=True
                                       )
        self.all_gt_boxes = {cls: [] for cls in self.class2idx}
        self.all_pred_boxes = {cls: [] for cls in self.class2idx}

    def inference(self, model, inputs):
        image, gt = inputs

        outputs = model(image)
        return outputs, gt

    def collection(self, item):
        outputs, gt = item
        outputs = self.postprocess(outputs)
        for i, one_image_boxes in enumerate(outputs):
            if one_image_boxes is None:
                for boxes in self.all_pred_boxes.values():
                    boxes.append({})
                continue
            for cls_name, cls_idx in self.class2idx.items():
                mask = one_image_boxes[..., -1] == cls_idx
                if np.any(mask) == 0:
                    self.all_pred_boxes[cls_name].append({})
                else:
                    image_cls_boxes = {
                        "confidence": one_image_boxes[mask][:, -2],
                        "bboxes": one_image_boxes[mask][:, :4],
                    }
                    self.all_pred_boxes[cls_name].append(image_cls_boxes)
        for i, one_image_boxes in enumerate(gt):
            one_image_boxes = one_image_boxes[one_image_boxes.sum(-1) > 0].numpy()
            if not self.is_xyxy:
                cxcywh = one_image_boxes[:, 1:]
                one_image_boxes = np.stack([
                    one_image_boxes[:, 0],
                    cxcywh[:, 0] - cxcywh[:, 2] / 2,
                    cxcywh[:, 1] - cxcywh[:, 3] / 2,
                    cxcywh[:, 0] + cxcywh[:, 2] / 2,
                    cxcywh[:, 1] + cxcywh[:, 3] / 2,
                ], -1)
            for cls_name, cls_idx in self.class2idx.items():
                mask = one_image_boxes[..., 0] == cls_idx
                if np.any(mask) == 0:
                    self.all_gt_boxes[cls_name].append({})
                else:
                    image_cls_boxes = {
                        "labels": one_image_boxes[mask][:, 0],
                        "bboxes": one_image_boxes[mask][:, 1:],
                    }
                    self.all_gt_boxes[cls_name].append(image_cls_boxes)

    def evaluate(self):
        aps = eval_map(self.all_gt_boxes, self.all_pred_boxes, self.ovthresh)
        new_aps = {f"ap_{k}": v for k, v in aps.items()}
        new_aps["map"] = np.mean(list(new_aps.values()))
        return new_aps

    def reset(self):
        self.all_gt_boxes = {cls: [] for cls in self.class2idx}
        self.all_pred_boxes = {cls: [] for cls in self.class2idx}
