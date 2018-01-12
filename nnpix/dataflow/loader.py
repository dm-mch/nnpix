import cv2
import numpy as np
import copy

from tensorpack import ProxyDataFlow, AugmentImageComponents

from .fileflow import get_fileflow
from ..common import AttrDict, list_shape, parse_value
from .fileflow import get_fileflow
from .imgaug import augment, ImageAugmentorListProxy, NotSafeAugmentorList, CropFlow

__all__ = ['get_train_data']

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
