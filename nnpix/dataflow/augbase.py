from tensorpack.dataflow.imgaug import ImageAugmentor, AugmentorList
from common import parse_value

class ImageAugmentorListProxy(ImageAugmentor):
    """ Augment list of images with shared params with input augmentator """

    def __init__(self, augmentator, shared_params=True):
        assert isinstance(augmentator, ImageAugmentor)
        self._base_augmentator = augmentator
        self._shared_params = shared_params
        super(ImageAugmentorListProxy, self).__init__()


    def _single_augment(self, img, params):
        """ Single image augmentation """
        assert len(img.shape) == 3 or len(img.shape) == 2, img.shape
        return self._base_augmentator._augment(img, params)

    def _get_augment_params(self, img):
        if type(img) == list:
            img = img[0]
        return self._base_augmentator._get_augment_params(img)

    def _augment(self, img, params):
        if type(img) != list:
            return self._single_augment(img, params)
        r = []
        for i in range(len(img)):
            if not self._shared_params: # if not shared params - generate new params for each
                params = self._get_augment_params(img[i])
            r.append(self._single_augment(img[i], params))
        return r


class CfgImageAugmentor(ImageAugmentor):
    """ Augmentator with configure """

    def __init__(self, cfg):
        super(CfgImageAugmentor, self).__init__()
        cfg = self._get_params(cfg)
        self._init(cfg)

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self':
                    setattr(self, k, v)

    def _get_params(self, cfg):
        """
            Parse cfg and return dictionary
            Result will be setted as self attributes
            In child classes here we can check/set default params
            Required keys in result:
                value - main value(param) for augmentator
                dp_index - int or list or tuple for datapoint index for augmentation
         """
        return parse_value(cfg, dp_index=None)


# rewrite for remove assertion: assert img.ndim in [2, 3], img.ndim
class NotSafeAugmentorList(AugmentorList):
    """
    Augment by a list of augmentors
    """

    def _augment_return_params(self, img):
        #assert img.ndim in [2, 3], img.ndim

        prms = []
        for a in self.augs:
            img, prm = a._augment_return_params(img)
            prms.append(prm)
        return img, prms

    def _augment(self, img, param):
        #assert img.ndim in [2, 3], img.ndim
        for aug, prm in zip(self.augs, param):
            img = aug._augment(img, prm)
        return img
