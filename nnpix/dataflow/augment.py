import cv2
import numpy as np

from tensorpack.dataflow.imgaug import ImageAugmentor

class Resize(ImageAugmentor):

    def __init__(self, resize, interpolate=cv2.INTER_LANCZOS4):
        """ param resize can be:
            1) list or tuple with len 2 (min,max) - random resize from uniform range
            2) Float - resize on exactly value
        """
        self.interpolate = interpolate
        self.resize = resize
        self.random = False
        if type(resize) == list or type(resize) == tuple:
            assert len(resize) == 2, resize
            self.random = True
        else:
            self.resize = float(self.resize)
        super(Resize, self).__init__()

    def _get_augment_params(self, img):
        scale = self.resize
        if self.random:
            scale = self.rng.uniform(*self.resize)
        return (int(img.shape[1]//scale),int(img.shape[0]//scale))

    def _augment(self, img, new_size):
        #print("Resize input {} new_size {}".format(img.shape, new_size))
        return cv2.resize(img, new_size, self.interpolate)









