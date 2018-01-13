import os
import cv2
import shutil
import numpy as np

from tensorpack.utils.utils import get_rng

from .base import CfgDataFlow
from ...common import list_shape
from ...registry import DataFlowRegistry

__all__ = ['CropFlow', 'PrintImageFlow']


class CropFlow(CfgDataFlow):
    """ Crop multiple times from one image """

    def _get_params(self, crop_cfg, data_cfg):
        crop_cfg = super(CropFlow, self)._get_params(crop_cfg, data_cfg)
        print("CropFloe data_cfg", data_cfg)
        # crop size
        crop_cfg['size'] = data_cfg.batch_shape
        # how many crops to do from one image
        crop_cfg['number'] = crop_cfg.value
        assert type(data_cfg.inputs) == list
        # list of scale factors for crop size, If 0 - no crop
        crop_cfg['scales'] = crop_cfg.scales if crop_cfg.scales is not None else [1] * len(data_cfg.inputs)
        return crop_cfg

    def reset_state(self):
        super(CropFlow, self).reset_state()
        self.rng = get_rng(self)

    def _crop(self, img, x, y, size):
        if type(img) == list:
            crops = []
            for i in img:
                crops.append(self._crop(i, x, y, size))
            return crops
        assert isinstance(img, np.ndarray), img
        return img[y:y+size, x: x+size].copy()

    def _get_crops(self, dp):
        base_index = self.scales.index(1) # should contain 1
        base_shape = dp[base_index].shape if  type(dp[base_index]) != list else dp[base_index][0].shape # first img shape
        if base_shape[0] <= self.size or base_shape[1] <= self.size: # small size for crop
            print("Image too small", base_shape)
            return None
        x = self.rng.randint(0, base_shape[1] - self.size)
        y = self.rng.randint(0, base_shape[0] - self.size)
        result = []
        for i in range(len(dp)):
            if i < len(self.scales) and self.scales[i] > 0:
                s = self.scales[i]
                result.append(self._crop(dp[i], int(x * s), int(y * s), int(self.size * s)))
            else:
                result.append(copy.copy(dp[i]))
        return result

    def get_data(self):
        for dp in super(CropFlow, self).get_data():
            for n in range(self.number):
                print("crop:", list_shape(dp))
                yield self._get_crops(dp)


class PrintImageFlow(CfgDataFlow):
    """ For debug usage. Save flow images in folder """

    def _get_params(self, print_cfg, data_cfg):
        print_cfg = super(PrintImageFlow, self)._get_params(print_cfg, data_cfg)
        print_cfg['path'] = print_cfg.path or print_cfg.value
        print_cfg['each'] = print_cfg.each or 1
        print_cfg['clear'] = print_cfg.clear or False
        return print_cfg

    def __init__(self, ds, print_cfg, data_cfg):
        super(PrintImageFlow, self).__init__(ds, print_cfg, data_cfg)
        if self.clear and os.path.isdir(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path, exist_ok=True)

    def print(self, dp, index):
        for n,img in enumerate(dp):
            if type(img) == list:
                # try to concatinate list in one image
                if isinstance(img[0], np.ndarray) and img[0].ndim in [2,3]:
                    shape = img[0].shape
                    if np.alltrue([isinstance(i, np.ndarray) and i.shape == shape for i in img]):
                        img = np.concatenate(img, axis=1)
            if isinstance(img, np.ndarray) and img.ndim in [2,3]:
                cv2.imwrite(os.path.join(self.path, "%i_%i.png"%(index, n)), img)

    def get_data(self):
        for i, d in enumerate(super(PrintImageFlow, self).get_data()):
            if i % self.each == 0:
                self.print(d, i)
            yield d

DataFlowRegistry().update({'crop': CropFlow, 'print': PrintImageFlow})