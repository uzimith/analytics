from sklearn import svm
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.externals import joblib
import numpy as np

class LinearSVM:
    def __init__(self, name="tmp"):
        self.clf = None
        self.name = name

    def load(self):
        self.clf = joblib.load('model/linearsvm-%s.pkl' % self.name)
        self.scaler = joblib.load('model/scaler.pkl')

    def train(self, labels, erps):
        print "training..."
        self.scaler = MaxAbsScaler()
        self.scaler.fit(erps)
        erps = self.scaler.transform(erps)

        self.clf = svm.SVC(probability = True, kernel='linear', C=1)
        self.clf.fit(erps, labels)

        scores = cross_validation.cross_val_score(self.clf, erps, labels, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        joblib.dump(self.clf, 'model/linearsvm-%s.pkl' % self.name)

    def predict(self, labels, erps, pattern_num):
        probabilities = [[] for row in range(pattern_num)]
        for (erp, label) in zip(erps, labels):
            erp = self.scaler.transform([erp])[0]
            probabilities[label].append(self.clf.predict_proba([erp])[0][1])
        summary = [np.mean(p) for p in probabilities]
        print(summary)
        result = np.argmax(summary)
        if(not np.isnan(np.max(summary)) and np.max(summary) != np.min(summary)):
            result = result + 1
        return result
