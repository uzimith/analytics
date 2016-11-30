import convert.erp

from receive import Receive
import numpy as np
from itertools import chain
import scipy.io
import glob

class LoadmatMulti(Receive):
    def __init__(self, multi_filename, type, average=1):
        Receive.__init__(self, average=average)
        self.index = 0
        self.type = type

        filenames = glob.glob(multi_filename)
        print(filenames)
        self.users = len(filenames)
        matfiles = [scipy.io.loadmat(filename) for filename in filenames]
        grand_erps = [matfile["erps"] for matfile in matfiles]
        if self.type == "train":
            grand_labels = [matfile["target_label"][0] for matfile in matfiles]
        elif self.type == "predict":
            grand_labels = [matfile["stimuli_label"][0] for matfile in matfiles]

        self.grand_erps = list(chain.from_iterable(grand_erps))
        self.grand_labels = list(chain.from_iterable(grand_labels))

    def receive(self, skip=False):
        i = self.index
        if not skip:
            self.erps.append(self.grand_erps[i])
            self.labels.append(self.grand_labels[i])
        self.index += 1
