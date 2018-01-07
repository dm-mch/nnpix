from keras.layers import Input, Conv2D
from keras import backend as K
import tensorflow as tf

def create_input(cfg, input=None):
    if input is not None:
        shape = list(K.get_session().run(tf.shape(input)))
        print("Input shape", shape)
        return Input(batch_shape=shape, tensor=input)
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

        return Input(batch_shape=shape, name = cfg.name + '_input')


