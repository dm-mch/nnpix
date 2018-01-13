
__all__ = ['AugmentRegistry', 'DataFlowRegistry', 'ModelRegistry']

class Registry:
    """ Registry storage """

    def __init__(self, t):
        print("Registry.__init__", t)
        assert isinstance(t, type), t
        self._type = t
        self._reg = {}

    def __setitem__(self, key, value):
        assert issubclass(value, self._type), (value, self._type)
        assert isinstance(key, str), key
        self._reg[key] = value

    def __getitem__(self, key):
        return self._reg.get(key, None)

    def update(self, d):
        """ Multiple update """
        for k, v in d.items():
            self.__setitem__(k,v)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AugmentRegistry(Registry, metaclass=Singleton):
    def __init__(self):
        from nnpix.dataflow.imgaug.base import CfgImageAugmentor
        super(AugmentRegistry, self).__init__(CfgImageAugmentor)

class DataFlowRegistry(Registry, metaclass=Singleton):
    def __init__(self):
        from nnpix.dataflow.imgaug.base import CfgDataFlow
        super(DataFlowRegistry, self).__init__(CfgDataFlow)

class ModelRegistry(Registry, metaclass=Singleton):
    def __init__(self):
        from nnpix.nn.model import CfgModel
        super(ModelRegistry, self).__init__(CfgModel)



