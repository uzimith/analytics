import numpy as np
from operator import itemgetter
from itertools import groupby
import scipy.io

class Loadmat:
    def __init__(self, subject, session, type, channel_num=8, separate=False, normalize=False):
        self.labels = []
        self.erps = []
        self.mat = scipy.io.loadmat("../mat_files_4555/subject%s_section%d.mat" % (subject, session) )
        self.index = 0
        self.channel_num = channel_num
        self.is_separate = separate
        self.is_normalize = normalize
        self.type = type

    def receive(self):
        i = self.index
        if self.type == "train":
            erp = self.mat['erps'][i]
            label = self.mat['target_label'][i][0]
        if self.type == "predict":
            erp = self.mat['erps'][i]
            label = self.mat['stimuli_label'][i][0]
        if self.is_normalize:
            erp = self.normalize(erp)
        if self.is_separate:
            erp = self.separate(erp)
        self.erps.append(erp)
        self.labels.append(label)
        self.index += 1

    def fetch(self):
        labels = self.labels
        erps = self.erps
        self.labels = []
        self.erps = []
        return erps

    def save(self):
        pass # already saved

    def separate(self, erp):
        frame_length = len(erp) / self.channel_num
        return np.squeeze(np.reshape(erp, (self.channel_num, frame_length)))

    def normalize(self, erp):
        erp_of_channels = [x - np.mean(x) /  np.std(x) for x in self.separate(erp)]
        normalized_erp = np.array(erp_of_channels).flatten()
        return normalized_erp

    def group(self):
        data = zip(self.labels, self.erps)
        data.sort(key=itemgetter(0))
        self.labels = [[k for k,v in v] for k, v in groupby(data, key=itemgetter(0))]
        self.erps = [[v for k,v in v] for k, v in groupby(data, key=itemgetter(0))]
