import os
import shutil
import cv2
import numpy as np
import copy

from tensorpack.dataflow import ProxyDataFlow

__all__ = ['ReadFilesFlow', 'PrintImageFlow']

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

class PrintImageFlow(ProxyDataFlow):
    """ For debug usage. Save flow images in folder """

    def __init__(self, ds, path, each=1, clear=None):
        super(PrintImageFlow, self).__init__(ds)
        self.path = path
        self.each = each
        if clear and os.path.isdir(self.path):
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