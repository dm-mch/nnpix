import cv2
import numpy as np

from tensorpack.dataflow import ProxyDataFlow
from ...registry import AugmentRegistry

from .base import CfgImageAugmentor

__all__ = ['Resize', 'INTERPOLATION']

INTERPOLATION = {'lanczos': lambda: cv2.INTER_LANCZOS4,
               'bicubic': lambda: cv2.INTER_CUBIC,
               'linear': lambda: cv2.INTER_LINEAR,
               'nearest': lambda: cv2.INTER_NEAREST,
               'random': lambda: np.random.choice([cv2.INTER_LANCZOS4,cv2.INTER_CUBIC,cv2.INTER_LINEAR, cv2.INTER_NEAREST]) }


class RandomUniformValue(CfgImageAugmentor):
    """ Get value as float or list [min,max] for random uniform param generation """

    def __init__(self, cfg, data_cfg):
        """ cfg.value can be:
            1) list or tuple with len 2 (min,max) - random value from uniform range
            2) Float - augment on exactly value
        """
        super(RandomUniformValue, self).__init__(cfg, data_cfg)
        self.random = False
        if type(self.value) == list or type(self.value) == tuple:
            assert len(self.value) == 2, self.value
            self.random = True
        else:
            self.value = float(self.value)

    def _get_augment_params(self, img):
        value = self.value
        if self.random:
            value = self.rng.uniform(*self.value)
        return value


class Resize(RandomUniformValue):

    def _get_params(self, cfg, data_cfg):
        params = super(Resize, self)._get_params(cfg, data_cfg)
        params["interpolation"] = INTERPOLATION['lanczos'] if params.interpolation is None  else INTERPOLATION[params['interpolation']]
        params['resize'] = params['value']
        return params

    def _augment(self, img, scale):
        new_size = (int(img.shape[1]//scale),int(img.shape[0]//scale))
        #print("Resize input {} new_size {}".format(img.shape, new_size))
        return cv2.resize(img, new_size, interpolation=self.interpolation())


class NormalNoise(RandomUniformValue):

    def _get_params(self, cfg, data_cfg):
        params = super(NormalNoise, self)._get_params(cfg, data_cfg)
        # params.clip is [min,max] for clipping
        assert params.clip is None or (isinstance(params.clip, list) and len(params.clip) == 2)
        params['clip'] = params.clip
        return params

    def _augment(self, img, noise_std):
        img = img + self.rng.normal(0, noise_std, size=img.shape)
        if self.clip:
            np.clip(img, self.clip[0], self.clip[1], out=img)
        return img

class GaussBlurr(RandomUniformValue):

    def _augment(self, img, ksize):
        k = cv2.getGaussianKernel(int(ksize), 0.3*((ksize-1)*0.5 - 1) + 0.8)
        k = k.dot(k.T)
        return cv2.filter2D(img, -1, k)


class Normalize(CfgImageAugmentor):

    def _get_params(self, cfg, data_cfg):
        params = super(Normalize, self)._get_params(cfg, data_cfg)
        params['div'] = params.div or 1
        params['minus'] = params.minus or 0
        return params

    def _augment(self, img, _):
        return img/self.div - self.minus

# Registry all shared augmentation in one storage
AugmentRegistry().update({'resize': Resize,
                          'normal_noise': NormalNoise,
                          'blur': GaussBlurr,
                          'normalize': Normalize })









