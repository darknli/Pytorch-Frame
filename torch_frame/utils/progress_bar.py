import tqdm
from .dist_utils import is_dist_avail_and_initialized, get_rank


class ProgressBar:
    """tqdm基础上包了一层，主要兼容ddp的多卡"""
    def __init__(self, *args, **kwargs):
        if not is_dist_avail_and_initialized() or get_rank() == 0:
            self.pbar = tqdm.tqdm(*args, **kwargs)
        else:
            self.pbar = None

    def update(self, *args, **kwargs):
        if self.pbar is not None:
            self.pbar.update(*args, **kwargs)

    def close(self):
        if self.pbar is not None:
            self.pbar.close()

    def set_postfix(self, *args, **kwargs):
        if self.pbar is not None:
            self.pbar.set_postfix(*args, **kwargs)

    def set_description(self, *args, **kwargs):
        if self.pbar is not None:
            self.pbar.set_description(*args, **kwargs)

    def set_postfix_str(self, *args, **kwargs):
        if self.pbar is not None:
            self.pbar.set_postfix_str(*args, **kwargs)

    def set_description_str(self, *args, **kwargs):
        if self.pbar is not None:
            self.pbar.set_description_str(*args, **kwargs)