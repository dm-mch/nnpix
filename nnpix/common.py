""" Common utils """

class AttrDict(dict):
    """ Access dictionary value via attribute. Example d.some_key = 1 """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        # parse and convert all inner dict to AttrDict
        for k,v in self.items():
            # replace '-' -> '_'
            if '-' in k:
                k_ = k.replace('-','_')
                self[k_] = v
                del self[k]
                k = k_

            if type(v) == list:
                self._parse_list(v)
            if type(v) == dict:
                self[k] = AttrDict(v)

        self.__dict__ = self

    def _parse_list(self, l):
        """ Recursive parse list and convert all dict to AttrDict """
        for i in range(len(l)):
            if type(l[i]) == list:
                self._parse_list(l[i])
            elif type(l[i]) == dict:
                l[i] = AttrDict(l[i])
