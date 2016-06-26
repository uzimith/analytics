import numpy as np
from operator import itemgetter
from itertools import groupby
import scipy.io

class Loadmat:
    def __init__(self, subject, session, type):
        self.labels = []
        self.erps = []
        self.mat = scipy.io.loadmat("../mat_files/subject%s_section%d.mat" % (subject, session) )
        self.index = 0
        self.type = type

    def receive(self):
        i = self.index
        if self.type == "train":
            self.erps.append(self.mat['erps_2d'][i])
            self.labels.append(self.mat['target_label'][i][0])
        if self.type == "predict":
            self.erps.append(self.mat['erps_2d'][i])
            self.labels.append(self.mat['stimuli_label'][i][0])
        self.index += 1

    def fetch(self):
        labels = self.labels
        erps = self.erps
        self.labels = []
        self.erps = []
        return erps

    def save(self):
        pass # already saved

    def group(self):
        data = zip(self.labels, self.erps)
        data.sort(key=itemgetter(0))
        self.labels = [[k for k,v in v] for k, v in groupby(data, key=itemgetter(0))]
        self.erps = [[v for k,v in v] for k, v in groupby(data, key=itemgetter(0))]
