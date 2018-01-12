import numpy as np

from tensorpack.dataflow import ProxyDataFlow
from tensorpack.utils.utils import get_rng

from ...common import list_shape

__all__ = ['CropFlow']

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

