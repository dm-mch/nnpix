import argparse
import tensorflow as tf
import numpy as np

from pprint import pprint

from parser import Parser
from common import AttrDict

from nn.model import NNModel
from dataflow.loader import get_train_data

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, default=None, help="Input file with YAML experiment configuration")
args = parser.parse_args()

p = Parser(args.file)


exp = p.next_exp()

#m = NNModel(exp.models.gen.join(exp.common))
#m.summary()
#
# m2 = NNModel(exp.models.gen)
# m2.summary()

#m.save_yaml()
#m.save_weights()

# vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# print("Total VARS:", len(vars))
# for v in vars:
#     print(v.name)

def list_shape(input):
    if type(input) == list:
        r = []
        for a in input:
            r.append(list_shape(a))
        return r
    else:
        return input.shape

ds = get_train_data(exp.train.data.join(exp.common))
ds.reset_state()
itr = ds.get_data()
for i in range(10):
    b = next(itr)
    print(i, list_shape(b))
