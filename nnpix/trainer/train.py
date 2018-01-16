import tensorflow as tf
from tensorpack import ModelDesc, InputDesc, TrainConfig, SimpleTrainer, launch_train_with_config

from ..nn.layers import get_input_shape
from ..nn.model import get_model

__all__ = ['train']

class ModelWrapSingle(ModelDesc):

    def __init__(self, train_cfg, model_cfg, common_cfg):
        super(ModelWrapSingle, self).__init__()
        self.hr_shape = get_input_shape(common_cfg, input=False, batch_size=False)
        self.lr_shape = get_input_shape(common_cfg, input=True, batch_size=False)
        self._model_cfg = model_cfg
        self._common_cfg = common_cfg
        self.lr = train_cfg.lr
        assert self.lr, self.lr
        print("ModelWrapSingle shapes HR {} LR {}".format(self.hr_shape, self.lr_shape))

    def _get_inputs(self):
        # [HR, LR]
        return [InputDesc(tf.float32, self.hr_shape, 'Ihr'), InputDesc(tf.float32, self.lr_shape, 'Ihr')]

    def _build_graph(self, inputs):
        hr_target, lr_input = inputs
        self.model = get_model(self._model_cfg.join(self._common_cfg), input=lr_input)
        self.model.summary()
        print("cost shapes", hr_target.get_shape(), self.model.kmodel.outputs[0].get_shape())
        self.cost = tf.losses.mean_squared_error(hr_target, self.model.kmodel.outputs[0])

    def _get_optimizer(self):
        lr = tf.get_variable(
            'learning_rate', initializer=self.lr, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        return opt

def train(train_cfg, model_cfg, common_cfg, dataflow):
    epochs = train_cfg.epochs
    assert epochs, epochs
    epoch_size = train_cfg.epoch_size
    assert epoch_size, epoch_size
    config = TrainConfig(
       model=ModelWrapSingle(train_cfg, model_cfg, common_cfg),
       dataflow=dataflow,
       #data=my_inputsource, # alternatively, use a customized InputSource
       #callbacks=[...],    # some default callbacks are automatically applied
       # some default monitors are automatically applied
       steps_per_epoch=epoch_size,   # default to the size of your InputSource/DataFlow
       max_epoch=epochs
    )
    print("Create trainer")
    trainer = SimpleTrainer()
    print("Run train")
    launch_train_with_config(config, trainer)
