from ._get_logger import logger
from .hooks import *
from .lr_scheduler import LRWarmupScheduler
from .utils import HistoryBuffer, misc
from .trainer import Trainer, MetricStorage
from .ddp_trainer import DDPTrainer

#  关闭opencv的多线程防止pytorch死锁
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
