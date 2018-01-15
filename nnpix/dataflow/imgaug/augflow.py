import os
import cv2
import copy
import shutil
import numpy as np

from .base import CfgDataFlow
from .augment import INTERPOLATION
from ...common import list_shape
from ...registry import DataFlowRegistry

__all__ = ['CropFlow', 'PrintImageFlow']


class CropFlow(CfgDataFlow):
    """ Crop multiple times from one image """

    def _get_params(self, crop_cfg, data_cfg):
        crop_cfg = super(CropFlow, self)._get_params(crop_cfg, data_cfg)
        print("CropFloe data_cfg", data_cfg)
        # crop size
        crop_cfg['batch_shape'] = data_cfg.batch_shape
        # how many crops to do from one image
        crop_cfg['number'] = crop_cfg.value
        assert type(data_cfg.inputs) == list
        # list of scale factors for crop size, If 0 - no crop
        crop_cfg['scales'] = crop_cfg.scales if crop_cfg.scales is not None else [1, 1]
        return crop_cfg

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
        if base_shape[0] <= self.batch_shape or base_shape[1] <= self.batch_shape: # small size for crop
            print("Image too small", base_shape)
            return None
        x = self.rng.randint(0, base_shape[1] - self.batch_shape)
        y = self.rng.randint(0, base_shape[0] - self.batch_shape)
        result = []
        for i in range(len(dp)):
            if i < len(self.scales) and self.scales[i] > 0:
                s = self.scales[i]
                result.append(self._crop(dp[i], int(x * s), int(y * s), int(self.batch_shape * s)))
            else:
                result.append(copy.copy(dp[i]))
        return result

    def get_data(self):
        for dp in super(CropFlow, self).get_data():
            for n in range(self.number):
                #print("crop:", list_shape(dp))
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

class CopyFlow(CfgDataFlow):
    def _get_params(self, clone_cfg, data_cfg):
        clone_cfg = super(CopyFlow, self)._get_params(clone_cfg, data_cfg)
        clone_cfg['clone_index'] = clone_cfg.value or 0
        return clone_cfg

    def get_data(self):
        for i, d in enumerate(super(CopyFlow, self).get_data()):
            if type(d) != list:
                d = [d]
            d.append(d[self.clone_index].copy())
            yield d

def get_M(sx, sy, a):
    """ Transformation matrix for cv2.warpAffine """
    a = (a * np.pi) / 180.  # degree to radian
    return np.array([[np.cos(a), -np.sin(a), sx], [np.sin(a), np.cos(a), sy]])



class FakeMultiframe(CfgDataFlow):
    def _get_params(self, cfg, data_cfg):
        params = super(FakeMultiframe, self)._get_params(cfg, data_cfg)
        params['shift'] = params.shift or 0
        params['angle'] = params.angle or 0
        params['nonback_shift'] = params.nonback_shift or 0
        params['nonback_angle'] = params.nonback_angle or 0
        params["interpolation"] = INTERPOLATION['lanczos'] if params.interpolation is None  else INTERPOLATION[params['interpolation']]
        params['frames'] = data_cfg.frames
        if params.dp_index is None:
            params['dp_index'] = 1 # by default use for
        assert data_cfg.frames is not None and data_cfg.frames > 1
        return params

    def _make_multiframe(self, img):
        assert img.ndim in [2, 3]

        back_params = []
        result = []
        for n in range(self.frames):
            r = [0, 0, 0] # [shift_x, shift_y, angle]
            back_params.append([0,0,0])
            if self.shift:
                r[0] += self.rng.uniform(-self.shift, self.shift)
                r[1] += self.rng.uniform(-self.shift, self.shift)
                back_params[-1][0] = r[0]
                back_params[-1][1] = r[1]
            if self.angle:
                r[2] += self.rng.uniform(-self.angle, self.angle)
                back_params[-1][2] = r[2]
            # shift and angle which is not in backtransform
            if self.nonback_shift:
                r[0] += self.rng.uniform(-self.nonback_shift, self.nonback_shift)
                r[1] += self.rng.uniform(-self.nonback_shift, self.nonback_shift)
            if self.nonback_angle:
                r[2] += self.rng.uniform(-self.nonback_angle, self.nonback_angle)
            # do transformation
            result.append(cv2.warpAffine(img, get_M(*r), (img.shape[1], img.shape[0]), flags=self.interpolation()))
        return result, back_params

    def get_data(self):
        for d in super(FakeMultiframe, self).get_data():
            d[self.dp_index], back_params = self._make_multiframe(d[self.dp_index])
            # add to back of datapoint special dict for share params between dataflows
            if not isinstance(d[-1], dict) or not d[-1].get('custom_params', False) is True:
                d.append({'custom_params': True}) # key for find this dict in dp
            #print(back_params)
            d[-1]['fake_multiframe'] = back_params
            yield d

    def is_add_custom_params(self):
        return True

class BackTransform(CfgDataFlow):

    def _get_params(self, cfg, data_cfg):
        params = super(BackTransform, self)._get_params(cfg, data_cfg)
        params['scale'] = params.value or 1
        params["interpolation"] = INTERPOLATION['lanczos'] if params.interpolation is None  else INTERPOLATION[params['interpolation']]
        if params.dp_index is None:
            params['dp_index'] = 1 # by default use for
        return params

    def _back_transform(self, imgs, params):
        assert len(imgs) == len(params)
        res = []
        for i,img in enumerate(imgs):
            p = [-params[i][0]/self.scale, -params[i][1]/self.scale, -params[i][2]]
            res.append(cv2.warpAffine(img, get_M(*p), (img.shape[1], img.shape[0]), flags=self.interpolation()))
        return res

    def get_data(self):
        for d in super(BackTransform, self).get_data():
            if isinstance(d[-1], dict) and 'fake_multiframe' in d[-1]:
                d[self.dp_index] = self._back_transform(d[self.dp_index], d[-1]['fake_multiframe'])
            yield d


DataFlowRegistry().update({'crop': CropFlow,
                           'print': PrintImageFlow,
                           'copy': CopyFlow,
                           'fake_multiframe': FakeMultiframe,
                           'back_transform': BackTransform})