from receive import Receive
import socket
from struct import *
import numpy as np
from operator import itemgetter
from itertools import groupby
import scipy.io

class UDP(Receive):
    def __init__(self, subject, session, type, port = 8081, channel_num=8, average=1, logname=""):
        Receive.__init__(self, channel_num=8, average=average)
        self.host = socket.gethostbyname(socket.gethostname())
        self.port = port
        self.bind()
        self.subject = subject
        self.session = session
        self.type = type

        # to save all data
        self.all_erps = []
        self.all_labels = []
        self.logname =logname

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
        self.all_erps += self.erps
        self.all_labels += self.labels
        data = Receive.fetch(self)
        return data

    def save(self):
        filename = "log/mat/%s-sub%s-sec%s-%s" % (self.logname, self.subject, self.session, self.type)

        if self.type == "train":
            scipy.io.savemat(filename, {'erps': self.all_erps, 'target_label': self.all_labels})
        elif self.type == "predict":
            scipy.io.savemat(filename, {'erps': self.all_erps, 'stimuli_label': self.all_labels})

        print("saved: %s" % filename)
