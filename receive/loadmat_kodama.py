from receive import Receive
import numpy as np
from operator import itemgetter
from itertools import groupby
import scipy.io

class LoadmatKodama(Receive):
    def __init__(self, subject, session, type, channel_num=8, separate=False, normalize=False, average=1):
        Receive.__init__(self, channel_num=8, average=average)
        self.type = type
        self.subject = subject
        self.is_separate = separate
        self.session = session
        self.index = 1
        self.load()

    def receive(self):
        if(len(self.erps) == 0):
            self.load()

    def group(self):
        pass

    def load(self):
        if self.type == "train":
            erps = [[], []]
            erps[0] = scipy.io.loadmat("../mat/Feature_NonTarget_sub0%d_tri0%d_chAll.mat" % (self.subject, self.session))['NTcmdAll']
            erps[1] = scipy.io.loadmat("../mat/Feature_Target_sub0%d_tri0%d_chAll.mat" % (self.subject, self.session))['TcmdAll']
        if self.type == "predict":
            mat = scipy.io.loadmat("../mat/Feature_Sorted_sub0%d_tri0%d_chAll.mat" % (self.subject, self.session))
            erps = mat['Scmd0%d' % self.index]
            erps = zip(*[iter(erps)]*10)
            self.index += 1
        if self.is_separate:
            erps = [[self.separate(erp) for erp in grouped_erps] for grouped_erps in erps]
        self.erps = erps

    def save(self):
        pass # already saved
