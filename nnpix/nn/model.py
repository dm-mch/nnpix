from keras.models import Model
from nnpix.nn import nntypes
import os
from keras.models import model_from_yaml
import tensorflow as tf

from .layers import PixelShuffle

MODELS_DIR = "./models/"

__all__ = ['CfgModel']

class CfgModel:
    """ Class for NN build from cfg """

    def __init__(self, cfg, input=None):
        self._cfg = cfg
        # If need, try to load from config keras yaml
        if cfg.load_yaml is not None:
            self.kmodel = self.load_yaml()
        else:
            # try to get constructor and build
            try:
                # find create function in nn.nntypes module by name
                constructor = getattr(nn.nntypes, 'nn_' + cfg.type)
            except AttributeError as e:
                raise NotImplementedError("NN type '{0}' not implemented. For support you need to add function 'nn_{0}'".format(cfg.type))
            self.kmodel = constructor(cfg, input) # keras model

        # load weights if needed
        if cfg.load_weights is not None:
            self.load_weights()

    def get_path(self, suffix=''):
        dir = os.path.join(MODELS_DIR, self._cfg.name)
        os.makedirs(dir, exist_ok=True)
        return os.path.join(dir, self._cfg.nn_name + suffix)

    # Keras model proxy functions
    # =======================================
    def load_yaml(self, file=None):
        file = self.get_path(suffix="_model.yaml") if file is None else file
        with open(file, 'r') as f:
            # in custom_object we should pass all custom layers/modules used in model
            self.kmodel = model_from_yaml(f.read(), custom_objects={'PixelShuffle': PixelShuffle, 'tf': tf})
            print("Model {} loaded from {}".format(self._cfg.name, file))
        return self.kmodel

    def save_yaml(self, file=None):
        file = self.get_path(suffix="_model.yaml") if file is None else file
        s = self.kmodel.to_yaml()
        with open(file, 'w') as f:
            f.write(s)


    def load_weights(self, file=None, by_name=True):
        file = self.get_path(suffix="_weights.hd5") if file is None else file
        self.kmodel.load_weights(file, by_name)
        print("Model {} weights loaded from {}".format(self._cfg.name, file))

    def save_weights(self, file=None):
        file = self.get_path(suffix="_weights.hd5") if file is None else file
        self.kmodel.save_weights(file)

    def summary(self):
        self.kmodel.summary()

    @property
    def layers(self):
        return self.kmodel.layers
    # =======================================
