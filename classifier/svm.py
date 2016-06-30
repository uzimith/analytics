import convert.erp

from sklearn import svm
from sklearn import cross_validation, grid_search
import sklearn.feature_selection as fs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.externals import joblib
import numpy as np

class SVM:
    def __init__(self, name="tmp", decimate=4):
        self.clf = None
        self.scaler = None
        self.name = name
        self.factor = decimate

    def load(self):
        self.clf = joblib.load('model/rbfsvm-%s.pkl' % self.name)

    def train(self, labels, erps):
        erps = convert.erp.decimate(erps, self.factor)

        clf = grid_search.GridSearchCV(svm.SVC(probability = True), parameters, n_jobs=4, cv=5)

        clf.fit(erps, labels)

        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))

        print(clf.best_params_)

        self.clf = clf.best_estimator_

        joblib.dump(self.clf, 'model/rbfsvm-%s.pkl' % self.name)

    def predict(self, labels, erps, pattern_num):
        erps = convert.erp.decimate(erps, self.factor)
        probabilities = [[] for row in range(pattern_num)]
        for (erp, label) in zip(erps, labels):
            probabilities[label].append(self.clf.predict_proba([erp])[0][1])
        summary = [np.mean(p) for p in probabilities]
        # print(summary)
        result = np.argmax(summary)
        if(not np.isnan(np.max(summary)) and np.max(summary) != np.min(summary)):
            result = result + 1
        return result
