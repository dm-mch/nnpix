from keras.models import Model
import nn.nntypes


class NNModel:
    """ Class for NN build from cfg """

    def __init__(self, cfg, input=None):
        constructor = getattr(nn.nntypes, 'nn_' + cfg.type)
        self.model = constructor(cfg, input)
