import cv2
import numpy as np
import copy

from tensorpack import ProxyDataFlow, AugmentImageComponents
from tensorpack.utils.utils import get_rng

from dataflow.fileflow import get_fileflow
from common import parse_value
from dataflow.fileflow import get_fileflow
from dataflow.augbase import ImageAugmentorListProxy, NotSafeAugmentorList
import dataflow.augment as augment
from common import AttrDict, list_shape


class ReadFilesFlow(ProxyDataFlow):

    def read(self, filename):
        #print("ReadFilesFlow.read", filename)
        return cv2.imread(filename)
        #if img_bgr is not None:
        #    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def list_read(self, files):
        r = []
        for f in files:
            if type(f) == str:
                r.append(self.read(f))
            elif type(f) == list:
                r.append(self.list_read(f))
            else:
                print("WARNING: Unsupported (not str and list)file name type {} for file {}".format(type(f), f))
                return None
            if r[-1] is None:
                print("WARNING: Can not read file {}".format(f))
                return None
        return r

    def get_data(self):
        for files in self.ds.get_data():
            if type(files) == str: files = [files]
            yield self.list_read(files)

class CropFlow(ProxyDataFlow):
    """ Crop multiple times from one image """

    def __init__(self, ds, size, number=1, scales=[1,1]):
        """
            ds - input dataflow
            size - crop size
            number - how many crops to do from one image
            scales -list of scale factors for crop size, If 0 - no crop
        """
        super(CropFlow, self).__init__(ds)
        self.size = size
        self.number = number
        self.scales = scales

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


def get_crop_ds(data_cfg, crop_cfg):
    def wrap(ds, data_cfg=data_cfg, crop_cfg=crop_cfg):
        size = data_cfg.batch_shape
        crop_cfg = parse_value(crop_cfg)
        number =  crop_cfg.value
        assert type(data_cfg.inputs) == list
        scales = crop_cfg.scales if crop_cfg.scales is not None else [1] * len(data_cfg.inputs)
        return CropFlow(ds, size, number, scales=scales)
    return wrap

def get_train_data(cfg, endless=True):
    ds_imgs = ReadFilesFlow(get_fileflow(cfg, endless=endless))

    if cfg.aug:
        assert type(cfg.aug) == list,type(cfg.aug)

        augs = []
        crop_ds_constructor = None
        for aug in cfg.aug:
            name = aug
            value = None
            if isinstance(aug, dict):
                assert len(aug) == 1, aug
                name = list(aug.keys())[0]
                value = aug[name]
            if name == 'resize':
                augs.append(ImageAugmentorListProxy(augment.Resize(value), shared_params=True))
            if name == 'crop':
                crop_ds_constructor = get_crop_ds(cfg, value)

        print("Agmentators", len(augs))
        ds_imgs = AugmentImageComponents(ds_imgs, NotSafeAugmentorList(augs), index=list(range(len(cfg.inputs))))

        if crop_ds_constructor:
            ds_imgs = crop_ds_constructor(ds_imgs)

    return ds_imgs
