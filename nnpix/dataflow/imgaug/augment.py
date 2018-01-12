import cv2
import numpy as np

from tensorpack.dataflow import ProxyDataFlow

from .base import CfgImageAugmentor

__all__ = ['Resize']

INTERPOLATION = {'lanczos': lambda: cv2.INTER_LANCZOS4,
               'bicubic': lambda: cv2.INTER_CUBIC,
               'linear': lambda: cv2.INTER_LINEAR,
               'nearest': lambda: cv2.INTER_NEAREST,
               'random': lambda: np.random.choice([cv2.INTER_LANCZOS4,cv2.INTER_CUBIC,cv2.INTER_LINEAR, cv2.INTER_NEAREST]) }

class Resize(CfgImageAugmentor):

    def _get_params(self, cfg):
        params = super(Resize, self)._get_params(cfg)
        if 'interpolation' not in params:
            params["interpolation"] = INTERPOLATION['lanczos']
        else:
            assert params['interpolation'] in INTERPOLATION, params['interpolation']
            params["interpolation"] = INTERPOLATION[params['interpolation']]
        return params

    def __init__(self, cfg):
        """ cfg.value can be:
            1) list or tuple with len 2 (min,max) - random resize from uniform range
            2) Float - resize on exactly value
        """
        super(Resize, self).__init__(cfg)
        self.resize = self.value
        self.random = False
        if type(self.resize) == list or type(self.resize) == tuple:
            assert len(self.resize) == 2, self.resize
            self.random = True
        else:
            self.resize = float(self.resize)


    def _get_augment_params(self, img):
        scale = self.resize
        if self.random:
            scale = self.rng.uniform(*self.resize)
        return (int(img.shape[1]//scale),int(img.shape[0]//scale))

    def _augment(self, img, new_size):
        print("Resize input {} new_size {}".format(img.shape, new_size))
        return cv2.resize(img, new_size, interpolation=self.interpolation())











