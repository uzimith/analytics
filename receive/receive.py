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

    def separate(self, erp):
        frame_length = len(erp) / self.channel_num
        return np.squeeze(np.reshape(erp, (self.channel_num, frame_length)))

    def combine(self, erp):
        return erp.flatten

    # non-good way
    def normalize(self, erp):
        erp_of_channels = [x - np.mean(x) /  np.std(x) for x in self.separate(erp)]
        normalized_erp = np.array(erp_of_channels).flatten()
        return normalized_erp

    def group(self):
        data = zip(self.labels, self.erps)
        data.sort(key=itemgetter(0))
        self.labels = [[k for k,v in v] for k, v in groupby(data, key=itemgetter(0))]
        self.erps = [[v for k,v in v] for k, v in groupby(data, key=itemgetter(0))]

    def undersampling(self, block_num, method="cosine", far=30):
        if method == "euclidean":
            target_erp = np.average(self.erps[1], axis=0)
            self.erps[0].sort(key=(lambda erp, target_erp=target_erp: np.linalg.norm(target_erp - erp)) )
            self.erps[0] = self.erps[0][far:far+block_num]
        if method == "cosine":
            non_target_erp = np.average(self.erps[0], axis=0).flatten()
            # self.erps[0].sort(key=(lambda erp, non_target_erp=non_target_erp: dis.cosine(non_target_erp, erp.flatten())) )
            self.erps[0] = sorted(self.erps[0], key=(lambda erp, non_target_erp=non_target_erp: dis.cosine(non_target_erp, erp.flatten())) )
            self.erps[0] = self.erps[0][far:far+block_num]
        if method == "cosine_more":
            # unuseful
            target_erp = np.average(self.erps[1], axis=0).flatten()
            non_target_erp = np.average(self.erps[0], axis=0).flatten()
            self.erps[0].sort(key=(lambda erp, non_target_erp=non_target_erp: dis.cosine(non_target_erp, erp.flatten())) )
            self.erps[0] = self.erps[0][far:far+block_num-10]
            self.erps[1].sort(key=(lambda erp, target_erp=non_target_erp: dis.cosine(target_erp, erp.flatten())) )
            self.erps[1] = self.erps[1][far:far+block_num-10]
        if method == "random":
            self.erps[0] = random.sample(self.erps[0], block_num)

    def average(self):
        self.erps = [[np.average(erp, axis=0) for erp in zip(*[iter(grouped_erps)]*self.group_by)] for grouped_erps in self.erps]

