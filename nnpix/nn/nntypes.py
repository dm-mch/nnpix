from keras.models import Model
from keras.layers import Input, Conv2D, Add

from nn.layers import create_input, conv2d_bn, residual_block, upscale_pixelshuffle
from common import extend

__all__ = ['nn_resnet']


def nn_resnet(cfg, input=None):
    """ ResNet based generator """

    input = create_input(cfg, input)
    print("Input shape:", input.shape)

    filters = extend(cfg.filters)
    # first convolution
    x = conv2d_bn(next(filters), cfg.first_conv_kernel or 3, padding='same', activation=cfg.activationg,
                 bn=cfg.bn, name=cfg.name+"_conv0")(input)
    # bottleneck for reduce filter dimension
    if cfg.bottleneck:
        x = conv2d_bn(next(filters), 3, padding='same', activation=cfg.activationg,
                      bn=cfg.bn, name=cfg.name + "_conv_bttlnck")(x)
    skip = x
    # Main residual blocks
    for i in range(cfg.blocks):
        x = residual_block(next(filters), 3, padding='same', activation=cfg.activationg,
                           bn=cfg.bn, name=cfg.name + "_block%i"%(i+1))(x)

    x = Add()([x, skip])
    x = conv2d_bn(next(filters), 3, strides=(1, 1), padding='same', activation=cfg.activationg,
                 bn=cfg.bn, name=cfg.name+"_convsum")(x)

    out = upscale_pixelshuffle(cfg.channels, 3, zoom=cfg.zoom, name=cfg.name+"_convps")(x)

    return Model(inputs=input, outputs=out)

