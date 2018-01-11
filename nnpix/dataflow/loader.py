import cv2
import numpy as np

from tensorpack import ProxyDataFlow, AugmentImageComponents
from dataflow.fileflow import get_fileflow

from dataflow.fileflow import get_fileflow
from dataflow.augbase import ImageAugmentorListProxy, NotSafeAugmentorList
import dataflow.augment as augment
from common import AttrDict


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


def get_train_data(cfg, endless=True):
    ds_imgs = ReadFilesFlow(get_fileflow(cfg, endless=endless))

    if cfg.aug:
        assert type(cfg.aug) == list,type(cfg.aug)

        augs = []
        for aug in cfg.aug:
            name = aug
            value = None
            if isinstance(aug, dict):
                assert len(aug) == 1, aug
                name = list(aug.keys())[0]
                value = aug[name]
            if name == 'resize':
                augs.append(ImageAugmentorListProxy(augment.Resize(value), shared_params=True))

    print("Agmentators", len(augs))

    ds_augs = AugmentImageComponents(ds_imgs, NotSafeAugmentorList(augs), index=list(range(len(cfg.inputs))))
    return ds_augs
