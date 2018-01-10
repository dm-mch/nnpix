import numpy as np
import cv2
import os
import re
import glob

from tensorpack.dataflow import *

class NoFilesError(Exception):
    """ When no file found in provided folder """
    pass


class EndlessListFlow(DataFromList):
    """ Endless dataflow through list """

    def __init__(self, lst, shuffle=True, endless=True):
        self.endless = endless
        super(EndlessListFlow, self).__init__(lst,shuffle)

    def get_data(self):
        if self.endless:
            while True:
                for i in super(EndlessListFlow, self).get_data():
                    yield i
        else:
            for i in super(EndlessListFlow, self).get_data():
                yield i


class FilesNameFlow(EndlessListFlow):
    """ DataFlow which iterate through file names in folder_tmpl """

    def __init__(self, folder_tmpl, **argw):
        self.files = glob.glob(folder_tmpl)
        if len(self.files) == 0: raise NoFilesError("No files found in %s"%folder_tmpl)
        super(FilesNameFlow, self).__init__(self.files, **argw)

class PairFilesNameFlow(EndlessListFlow):
    """
        DataFlow which iterate through file names in 2 folders
        All files not existed in both folders are excluded (only intersection used)
        TODO: Support recursive for nested folders...
    """

    def __init__(self, folder_tmpl1,  folder_tmpl2, **argw):
        self.files1 = glob.glob(folder_tmpl1)
        self.files2 = glob.glob(folder_tmpl2)
        if len(self.files1) == 0: raise NoFilesError("No files found in %s"%folder_tmpl1)
        if len(self.files2) == 0: raise NoFilesError("No files found in %s" % folder_tmpl2)
        # TODO: Support intersection with subfolders
        intersection = set(map(os.path.basename, self.files1)).intersection(set(map(os.path.basename, self.files2)))
        path1 = os.path.dirname(self.files1[0])
        path2 = os.path.dirname(self.files2[0])
        self.files = list(map(lambda f: [os.path.join(path1, f), os.path.join(path2, f)], intersection))
        super(PairFilesNameFlow, self).__init__(self.files, **argw)

class PairMultiframeFlow(EndlessListFlow):

    def __init__(self, folder_single,  folder_multi, frames=4, **argw):
        """ folder_multi should contain files with name foder_single_name_X.ext where X from 0 to frames-1 """
        self.files_single = glob.glob(folder_single)
        self.files_multi = glob.glob(folder_multi)
        if len(self.files_single) == 0: raise NoFilesError("No files found in %s"%folder_single)
        if len(self.files_multi) == 0: raise NoFilesError("No files found in %s" % folder_multi)
        # TODO: Support intersection with subfolders
        def basename(path): # find base filename except mutiframe index: name.png for name_0.png
            b = os.path.basename(path)
            b = re.findall(r"^(.+)_[0-9]+\.(.+)$", b)
            if len(b) == 1: return b[0][0] + '.' + b[0][1]

        def multiframe(f, count): # for name return all multiframes: name.png -> [name_0.png ... name_COUNT.png]
            base = os.path.splitext(f)
            return [base[-2] + "_%i"%i + base[1] for i in range(count)]

        intersection = set(map(os.path.basename, self.files_single)).intersection(set(map(basename, self.files_multi)))
        path_single = os.path.dirname(self.files_single[0])
        path_multi = os.path.dirname(self.files_multi[0])
        multi_all_set = set(map(os.path.basename, self.files_multi))
        all_frames_exist = filter(lambda f: np.alltrue(i in multi_all_set for i in multiframe(f, frames)), intersection)
        self.files = list(map(lambda f: [os.path.join(path_single, f), multiframe(os.path.join(path_multi, f), frames)], all_frames_exist))
        super(PairMultiframeFlow, self).__init__(self.files, **argw)



if __name__ == "__main__":
    ds = PairMultiframeFlow(r"/media/dimakl/Data hdd 1/datasets/div2k/DIV2K_valid_HR/*.png",r"/media/dimakl/Data hdd 1/datasets/div2k/DIV2K_valid2/*.png", frames=2, endless=False)
    ds.reset_state()
    itr = ds.get_data()
    for i in range(2000):
        print(i, next(itr))
