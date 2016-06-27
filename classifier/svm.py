from sklearn import svm
from sklearn import cross_validation, grid_search
from sklearn.externals import joblib
import numpy as np

class SVM:
    def __init__(self):
        self.clf = None

    def load(self):
        self.clf = joblib.load('model/clf.pkl')

    def train(self, labels, erps):
        print "training..."
        # parameters = [{'kernel': ['rbf'], 'gamma': [0.02564103, 1e-1, 1e-2, 1e-3, 1e-4],
        #                      'C': [1, 10, 100, 1000]}]
        # parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
        #                      'C': [1, 10, 100, 1000]},
        #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        # svr = svm.SVC(probability = True)
        # clf = grid_search.GridSearchCV(svr, parameters, n_jobs=4, cv=5)
        #
        # clf.fit(erps, labels)
        #
        # for params, mean_score, scores in clf.grid_scores_:
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean_score, scores.std() * 2, params))
        #
        # print(clf.best_params_)
        #
        # self.clf = clf.best_estimator_

        self.clf = svm.SVC(kernel= 'rbf', gamma = 0.02564103, probability = True)
        self.clf.fit(erps, labels)

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
