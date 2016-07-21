import convert.erp
import data

from stepwise import stepwisefit

from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.externals import joblib
import numpy as np

import pprint

class SWLDA:
    def __init__(self, name="tmp", decimate=5):
        self.clf = None
        self.factor = decimate
        self.name = name
        self.index = None
        self.frame_length = 0

    def load(self):
        self.clf = joblib.load('model/swlda-%s.pkl' % self.name)
        self.index = np.load("model/swlda-index-%s.npy" % self.name)

    def train(self, labels, erps):
        if self.factor != 1:
            erps = convert.erp.decimate(erps, self.factor)

        self.frame_length = len(erps[0]) / 8

        ( b, se, pval, inmodel, stats, nextstep, history ) = stepwisefit( erps, labels, maxiter = 60, penter = 0.1, premove = 0.15)
        self.index = inmodel
        erps = [np.array(erp)[self.index] for erp in erps]
        self.clf = lda()
        self.clf.fit(erps, labels)

        scores = cross_validation.cross_val_score(self.clf, erps, labels, cv=6)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        self.show_feature()

        joblib.dump(self.clf, 'model/swlda-%s.pkl' % self.name)
        np.save("model/swlda-index-%s.npy" % self.name, self.index)

    def show_feature(self):
        print("flame: %d, selected features: (%d)" % (self.frame_length, len(np.where(self.index)[0])))
        # print(" ".join([ "%s" % data.electrodes()[index / self.frame_length] for index in np.where(self.index)[0]]))
        print(" ".join([ "%s[%d]" % (data.electrodes()[index / self.frame_length], index % self.frame_length) for index in np.where(self.index)[0]]))

    def predict(self, labels, erps, pattern_num):
        if self.factor != 1:
            erps = convert.erp.decimate(erps, self.factor)
        erps = [np.array(erp)[self.index] for erp in erps]

        probabilities = [[] for row in range(pattern_num)]
        for (erp, label) in zip(erps, labels):
            probabilities[label].append(self.clf.predict_proba([erp])[0][1])
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(probabilities)
        summary = [np.mean(p) for p in probabilities]
        print(summary)
        result = np.argmax(summary)
        if(not np.isnan(np.max(summary)) and np.max(summary) != np.min(summary)):
            result = result + 1
        return result
