import cv2
import numpy as np

from tensorpack.dataflow import ProxyDataFlow

from ..common import list_shape

__all__ = ['ReadFilesFlow', 'RemoveCustomParamsFlow', 'ListToNumpyFlow', 'EndlessData', 'PrintShape']

class ReadFilesFlow(ProxyDataFlow):

    def read(self, filename):
        #print("ReadFilesFlow.read", filename)
        return cv2.imread(filename)
        #if img_bgr is not None:
        #    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def list_read(self, files):
        r = []
        for f in files:
            if type(f) == str:
                r.append(self.read(f))
            elif type(f) == list:
                r.append(self.list_read(f))
            else:
                print("WARNING: Unsupported (not str and list)file name type {} for file {}".format(type(f), f))
                return None
            if r[-1] is None:
                print("WARNING: Can not read file {}".format(f))
                return None
        return r

    def get_data(self):
        for files in self.ds.get_data():
            if type(files) == str: files = [files]
            yield self.list_read(files)


class EndlessData(ProxyDataFlow):
    """ Take data points from another DataFlow and produce them until
        it's exhausted for certain amount of times. i.e.:
        dp1, dp2, .... dpn, dp1, dp2, ....dpn
    """

    def __init__(self, ds, nr=-1):
        """
        Args:
            ds (DataFlow): input DataFlow
            nr (int): number of times to repeat ds.
                Set to -1 to repeat ``ds`` infinite times.
        """
        self.nr = nr
        super(EndlessData, self).__init__(ds)

    def size(self):
        return self.ds.size() # NOT AN REAL SIZE, BECAUSE REPEATED

    def get_data(self):
        if self.nr == -1:
            while True:
                for dp in self.ds.get_data():
                    yield dp
        else:
            for _ in range(self.nr):
                for dp in self.ds.get_data():
                    yield dp


class RemoveCustomParamsFlow(ProxyDataFlow):
    """ Remove last dict with custom params from datapoint """
    def get_data(self):
        for d in super(RemoveCustomParamsFlow, self).get_data():
            if isinstance(d[-1], dict) and 'custom_params' in d[-1]:
                d = d[:-1]
            yield d

class ListToNumpyFlow(ProxyDataFlow):
    """ Merge list to one numpy array """
    def __init__(self, ds, index):
        super(ListToNumpyFlow, self).__init__(ds)
        self.index = index

    def get_data(self):
        for d in super(ListToNumpyFlow, self).get_data():
            d[self.index] = np.array(d[self.index])
            yield d


class PrintShape(ProxyDataFlow):
    """ For debug only: Print shapes of datapoints """
    def get_data(self):
        for d in super(PrintShape, self).get_data():
            print("PrintShape:", list_shape(d))
            yield d
