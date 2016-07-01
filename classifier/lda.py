from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.externals import joblib
import numpy as np

class LDA:
    def __init__(self, name="tmp"):
        self.clf = None

    def load(self):
        self.clf = joblib.load('model/lda.pkl')

    def train(self, labels, erps):
        print "training..."

        self.clf = lda()
        self.clf.fit(erps, labels)

        scores = cross_validation.cross_val_score(self.clf, erps, labels, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        joblib.dump(self.clf, 'model/lda.pkl')

    def predict(self, labels, erps, pattern_num):
        probabilities = [[] for row in range(pattern_num)]
        for (erp, label) in zip(erps, labels):
            probabilities[label].append(self.clf.predict_proba([erp])[0][1])
        summary = [np.mean(p) for p in probabilities]
        print(summary)
        result = np.argmax(summary)
        if(not np.isnan(np.max(summary)) and np.max(summary) != np.min(summary)):
            result = result + 1
        return result
