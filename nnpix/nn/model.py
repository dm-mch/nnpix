import os
from keras.models import model_from_yaml
import tensorflow as tf

from .layers import PixelShuffle
from ..registry import ModelRegistry

MODELS_DIR = "./models/"

__all__ = ['CfgModel', 'get_model']

def get_model(cfg, input=None):
    models = ModelRegistry()
    if models[cfg.type] is not None:
        return models[cfg.type](cfg, input=input)
    else:
        raise Exception("Model '{}' not found".format(cfg.type))

class CfgModel:
    """ Class for NN build from cfg """

    def __init__(self, cfg, input=None):
        self._cfg = cfg
        # If need, try to load from config keras yaml
        if cfg.load_yaml is not None:
            self.kmodel = self.load_yaml()
        else:
            self.kmodel = self.create(cfg, input) # keras model

        # load weights if needed
        if cfg.load_weights is not None:
            self.load_weights()

    def creat(self, cfg, input=None):
        """ Constructor of keras model """
        # return keras.models.Model()
        raise NotImplementedError("Should be implemented in child classes")

    def get_path(self, suffix=''):
        root = self._cfg.models_dir or MODELS_DIR
        dir = os.path.join(root, self._cfg.name)
        os.makedirs(dir, exist_ok=True)
        return os.path.join(dir, self._cfg.nn_name + suffix)

    # Keras model proxy functions
    # =======================================
    def load_yaml(self, file=None, custom_objects=None):
        file = self.get_path(suffix="_model.yaml") if file is None else file
        with open(file, 'r') as f:
            # in custom_object we should pass all custom layers/modules used in model
            co = {'PixelShuffle': PixelShuffle, 'tf': tf}
            if custom_objects is not None:
                co.update(custom_objects)
            self.kmodel = model_from_yaml(f.read(), custom_objects=co)
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
