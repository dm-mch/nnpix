import argparse
import tensorflow as tf
import numpy as np
import time

from pprint import pprint

from nnpix.parser import Parser
from nnpix.common import AttrDict, list_shape

from nnpix.nn.model import get_model
from nnpix.dataflow import get_train_data


parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, default=None, help="Input file with YAML experiment configuration")
args = parser.parse_args()

p = Parser(args.file)


exp = p.next_exp()

pprint((exp.train.data))

#m = get_model(exp.models.gen.join(exp.common))
#m.summary()
#
#m2 = get_model(exp.models.gen.join(exp.common))
#m2.summary()

#m.save_yaml()
#m.save_weights()

# vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# print("Total VARS:", len(vars))
# for v in vars:
#     print(v.name)


t = time.time()
ds = get_train_data(exp.train.data, exp.common)
ds.reset_state()
itr = ds.get_data()
for i in range(1000):
    b = next(itr)
    print(i, list_shape(b))

print("Total time", time.time() - t)
