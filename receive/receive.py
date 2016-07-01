from operator import itemgetter
from itertools import groupby

import numpy as np
import scipy.spatial.distance as dis
import random

class Receive(object):
    def __init__(self, channel_num=8, average=1):
        self.erps = []
        self.labels = []
        self.channel_num = channel_num
        self.group_by = average

    def receive(self):
        pass

    def save(self):
        pass

    def clear(self):
        self.erps = []
        self.labels = []

    def fetch(self):
        self.average()
        labels = self.labels
        erps = self.erps
        self.labels = []
        self.erps = []
        return erps

    def group(self):
        data = zip(self.labels, self.erps)
        data.sort(key=itemgetter(0))
        self.labels = [[k for k,v in v] for k, v in groupby(data, key=itemgetter(0))]
        self.erps = [[v for k,v in v] for k, v in groupby(data, key=itemgetter(0))]

    def average(self):
        self.erps = [[np.average(erp, axis=0) for erp in zip(*[iter(grouped_erps)]*self.group_by)] for grouped_erps in self.erps]

