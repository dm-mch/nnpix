from keras.models import Model
from keras.layers import Input, Conv2D
from keras.layers.advanced_activations import PReLU
from nn.layers import create_input

__all__ = ['nn_resnet']


# def get_activation(name):
#     """ Return callable activatin. Example: Conv2D(activation=get_activation(name)())"""
#     if name in [None, 'relu', 'sigmoid']:
#         return lambda: name
#     elif name == 'prelu':
#         return lambda: PReLU()

def nn_resnet(cfg, input=None):

    input = create_input(cfg, input)
    print("Input shape:", input.shape)
    l = Conv2D(32,3, name=cfg.name + "_conv1")(input)
    out = Conv2D(cfg.channels,3, name=cfg.name + "_conv2")(l)

    return Model(inputs=input, outputs=out)

