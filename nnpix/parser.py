import yaml

from common import AttrDict

class Parser:
    """ Parser for YAML config """

    def __init__(self, file):
        self._file = file # file path
        self._current = 0  # current experiment index
        self._load() # read YAML


    def _load(self):
        with open(self._file, "r") as f:
            self._cfg = yaml.load(f)
        assert type(self._cfg) == list, "Type of root {} should be list, has: {}".format(self._file, type(self._cfg))
        print("Config {} loaded, contain {} experiments, current {}".format(self._file, len(self._cfg), self._current))
        return self._cfg

    def next_exp(self):
        """ Get configuration of next experiment """
        self._load() # reload config from file
        assert len(self._cfg) > self._current, "Len cfg %i, current %i"%(len(self._cfg), self._current)
        assert type(self._cfg[self._current]) == dict, "Experiment should be a dictionary"
        assert len(self._cfg[self._current].keys()) == 1, "Experiment should have only one kes, has: {}".format(self._cfg[self._current].keys())

        name = list(self._cfg[self._current].keys())[0]
        exp_cfg = self._cfg[self._current][name]
        self._current += 1

        return ExpConfig(name, exp_cfg)


class ExpConfig:

    # config sections name
    TRAIN = 'train'
    COMMON = 'common'
    VALID = 'validate'
    NN = 'network'

    def __init__(self, name, cfg, common=True):
        self.name = name
        self._cfg = cfg
        self.models = self._find_models()
        self.m = self.models # shortcut

        self.common = None if self.COMMON not in cfg.keys() else AttrDict(cfg[self.COMMON])
        self.c = self.common # shortcut

        self.train = None if self.TRAIN not in cfg.keys() else AttrDict(cfg[self.TRAIN])
        self.t = self.train # shortcut

        self.validate = None if self.VALID not in cfg.keys() else AttrDict(cfg[self.VALID])
        self.v = self.validate # shortcut

        # all sections except self.common
        self._sections = [self.train, self.validate] + list(self.models.values())

        if common:
            # Add common section to all
            for s in self._sections:
                s.update(self.common)

    def _find_models(self):
        """ Find elements where key name start from self.NN
            Try to split it by '-' and use right part as name
            Example: network-generator, name = generator
            return AttrDict with models configs """
        models = {}
        for k in self._cfg.keys():
            if type(k) == str and k.startswith(self.NN):
                name = k
                if '-' in k:
                    name = k.split('-')[1]
                models[name] = self._cfg[k]

                # if model has not attribute name - use default config name
                if models[name].get('nn_name', None) is None:
                    models[name]['nn_name'] = name

                # if model has not attribute name - use default config name
                if models[name].get('name', None) is None:
                    models[name]['name'] = self.name

        return AttrDict(models)
