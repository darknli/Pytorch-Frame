import numpy as np


__all__ = [
    "RepeatDataset",
]


# 源自https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/dataset_wrappers.py
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = getattr(dataset, 'PALETTE', None)
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def get_cat_ids(self, idx):
        """Get category ids of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.dataset.get_cat_ids(idx % self._ori_len)

    def get_ann_info(self, idx):
        """Get annotation of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.dataset.get_ann_info(idx % self._ori_len)

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len