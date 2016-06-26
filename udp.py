import socket
from struct import *
import numpy as np
from operator import itemgetter
from itertools import groupby

class UDP:
    def __init__(self, port = 8081):
        self.host = socket.gethostbyname(socket.gethostname())
        self.port = port
        self.bind()

        self.labels = []
        self.erps = []
        self.train_erps = []
        self.predict_erps = []

    def bind(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        print("udp : ", (self.host, self.port))

    def receive(self, skip = False):
        a = self.sock.recv(65535)
        if skip:
            return
        data = []
        for i in range(0, len(a), 4):
            tmp = unpack('f', a[i:i+4])
            data.append(tmp[0])
        label = int(data.pop())
        erp = np.asarray(data)
        self.labels.append(label)
        self.erps.append(erp)

    def fetch(self):
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

    def average(self, group_by):
        pass
