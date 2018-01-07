import argparse
from pprint import pprint

from parser import Parser

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, default=None, help="Input file with YAML experiment configuration")
args = parser.parse_args()

p = Parser(args.file)


exp = p.next_exp()
print(exp.common)
print(exp.train.lr)
print(exp.train.aug[1].fake_multiframe.shift)

pprint(exp.models)
print(exp.m.gen.batch_size)
print(exp.m.disc.batch_size)

