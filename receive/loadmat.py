import convert.erp

from receive import Receive
import numpy as np
from operator import itemgetter
from itertools import groupby
import scipy.io

class Loadmat(Receive):
    def __init__(self, subject, session, type, channel_num=8, separate=False, normalize=False, average=1, filename="../mat/512hz4555/sub%s_sec%s.mat"):
        Receive.__init__(self, channel_num=8, average=average)
        self.mat = scipy.io.loadmat(filename % (subject, session) )
        self.index = 0
        self.is_separate = separate
        self.is_normalize = normalize
        self.type = type
        self.other_labels = []

    def receive(self):
        i = self.index
        if self.type == "train":
            erp = self.mat['erps'][i]
            label = self.mat['target_label'][i][0]
            other_label = self.mat['stimuli_label'][i][0]
        if self.type == "predict":
            erp = self.mat['erps'][i]
            label = self.mat['stimuli_label'][i][0]
            other_label = self.mat['target_label'][i][0]
        if self.is_normalize:
            erp = self.normalize(erp)
        self.erps.append(erp)
        self.labels.append(label)
        self.other_labels.append(other_label)
        self.index += 1

    def group(self):
        data = zip(self.labels, self.erps, self.other_labels)
        data.sort(key=itemgetter(0))
        self.labels       = [[v2  for k, v1, v2 in v] for k, v in groupby(data, key=itemgetter(0))]
        self.erps         = [[v1 for k, v1, v2 in v] for k, v in groupby(data, key=itemgetter(0))]
        self.other_labels = [[k for k, v1, v2 in v] for k, v in groupby(data, key=itemgetter(0))]

    def fetch(self):
        data = Receive.fetch(self)
        self.other_labels = []
        return data

    def clear(self):
        Receive.clear(self)
        self.other_labels = []

    def save(self):
        pass # already saved
