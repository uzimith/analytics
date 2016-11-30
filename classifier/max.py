from sklearn import svm
from sklearn import cross_validation, grid_search
from sklearn.externals import joblib
import numpy as np

class SVM:
    def __init__(self):
        self.clf = None

    def load(self):
        pass

    def train(self, labels, erps):
        pass

    def predict(self, labels, erps, pattern_num):
        height = [[] for row in range(pattern_num)]
        for (erp, label) in zip(erps, labels):
            height[label].append(np.max(erp))
        pass
