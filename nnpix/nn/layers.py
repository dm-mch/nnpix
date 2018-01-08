from keras.layers import Input, Conv2D, BatchNormalization, PReLU, Activation, Add, Lambda, Layer
from keras import backend as K
import tensorflow as tf

class PixelShuffle(Layer):
    """ Layer with depth_to_space """

    def __init__(self, zoom=2, **kwargs):
        self.zoom = zoom
        super(PixelShuffle, self).__init__(**kwargs)

    def get_config(self):
        """ For load model. config will used as constructor params """
        config = {'zoom': self.zoom}
        config.update(super(PixelShuffle, self).get_config())
        return config

    def call(self, x):
        return tf.depth_to_space(x, self.zoom)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[3] % (self.zoom ** 2) == 0
        return (input_shape[0], input_shape[1] * self.zoom, input_shape[2] * self.zoom, input_shape[3] // (self.zoom ** 2))

def create_input(cfg, input=None):
    """ Return Input layer with right shape based on cfg or input tensor """

    name = cfg.name + '_input'
    if input is not None:
        shape = list(K.get_session().run(tf.shape(input)))
        print("Input shape", shape)
        return Input(batch_shape=shape, tensor=input, name=name)
    else:
        shape = [None, None, None, None]
        # for multiframe we need additional dimension

        if cfg.get('frames', 1) > 1:
            shape.append(None)
            shape[1] = cfg.frames

        if cfg.get('batch_size', None):
            shape[0] = cfg.batch_size

        if cfg.get('batch_shape', None):
            shape[-2] = cfg.batch_shape
            shape[-3] = cfg.batch_shape

        if cfg.get('channels', None):
            shape[-1] = cfg.channels

        return Input(batch_shape=shape, name = name)


def conv2d_bn(filters, kernel_size, strides=(1, 1), padding='same', activation=None, bn=False, name=None):
    """ Combine Conv2D, BatchNorm and Activation """

    def wrap(input):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, name=name)(input)
        if bn:
            x = BatchNormalization(name=name + '_bn')(x)
        if activation == "prelu":
            x = PReLU(name=name + '_prelu')(x)
        elif activation is not None:
            x = Activation(activation, name=name + "_" + activation)
        return x
    return wrap

def residual_block(filters, kernel_size, scale=0.5, padding='same', activation=None, bn=False, name=None):
    """ Residual block with 2 convolutions, skip and scale. Last convolution - no activation, no bn """

    def wrap(input):
        x = conv2d_bn(filters, kernel_size, padding=padding, activation=activation, bn=bn, name=name + '_res1')(input)
        x = conv2d_bn(filters, kernel_size, padding=padding, activation=None, bn=False, name=name + '_res2')(x)
        # sum with scale
        x = Add()([Lambda(lambda x, scale=scale: x * scale)(input), Lambda(lambda x, scale=scale: x * (1-scale))(x)])
        return x
    return wrap

def upscale_pixelshuffle(filters, kernel_size, zoom=2, activation=None, bn=False, name=None):
    """ Combine convolution and pixelshuffle """

    def wrap(input):
        x = conv2d_bn(filters * (zoom**2), kernel_size, padding='same', activation=activation, bn=bn, name=name)(input)
        x = PixelShuffle(zoom=zoom)(x)
        return x
    return wrap


