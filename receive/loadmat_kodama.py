from receive import Receive
import numpy as np
from operator import itemgetter
from itertools import groupby
import scipy.io

class LoadmatKodama(Receive):
    def __init__(self, subject, session, type, channel_num=8, average=1, folder="mat_500_D1_E1"):
        Receive.__init__(self, channel_num=8, average=average)
        self.type = type
        self.subject = subject
        self.session = session
        self.index = 1
        self.folder = folder
        self.load()

    def receive(self):
        if(len(self.erps) == 0):
            self.load()

    def group(self):
        pass

    def load(self):
        if self.type == "train":
            erps = [[], []]
            erps[0] = scipy.io.loadmat("../mat/mat_kodama/%s/Feature_NonTarget_sub0%s_tri0%s_chAll.mat" % (self.folder, self.subject, self.session))['NTcmdAll']
            erps[1] = scipy.io.loadmat("../mat/mat_kodama/%s/Feature_Target_sub0%s_tri0%s_chAll.mat" % (self.folder, self.subject, self.session))['TcmdAll']
        if self.type == "predict":
            mat = scipy.io.loadmat("../mat/mat_kodama/%s/Feature_Sorted_sub0%s_tri0%s_chAll.mat" % (self.folder, self.subject, self.session))
            erps = mat['Scmd0%s' % self.index]
            erps = zip(*[iter(erps)]*10)
            self.index += 1
        self.erps = erps

    def save(self):
        pass # already saved
