from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib
import numpy as np

class LinearSVM:
    def __init__(self):
        self.clf = None

    def load(self):
        self.clf = joblib.load('model/clf.pkl')

    def train(self, labels, erps):
        print "training..."
        self.clf = svm.SVC(probability = True, kernel='linear', C=1)

        self.clf.fit(erps, labels)

        scores = cross_validation.cross_val_score(self.clf, erps, labels, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        joblib.dump(self.clf, 'model/clf.pkl')

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
