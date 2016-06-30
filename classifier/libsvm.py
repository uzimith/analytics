from sklearn import svm
from sklearn import cross_validation, grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.externals import joblib
from svm import *
from svmutil import *
import numpy as np

class LibSVM:
    def __init__(self, scale=True):
        self.clf = None
        self.scaler = None
        self.scale = scale

    def load(self):
        self.model = svm_load_model('model/lsvm.model')
        self.scaler = joblib.load('model/lsvm-scaler.pkl')

    def train(self, labels, erps):
        print "training..."
        self.scaler = MaxAbsScaler()
        self.scaler.fit(erps)
        erps = self.scaler.transform(erps)
        problem = svm_problem(labels, erps.tolist())
        parameter = svm_parameter("-s 0 -t 0 -c 1 -g 0.25 -b 1")
        self.model = svm_train(problem, parameter)
        svm_save_model("model/lsvm.model", self.model)
        joblib.dump(self.scaler, 'model/lsvm-scaler.pkl')

    def predict(self, labels, erps, pattern_num):
        probabilities = [[] for row in range(pattern_num)]
        for (erp, label) in zip(erps, labels):
            erp = self.scaler.transform([erp])[0]
            _, acc, vals = svm_predict([0], [erp.tolist()], self.model, '-b 1 -q')
            if(len(vals[0]) == 2):
                probabilities[label].append(vals[0][1])
        summary = [np.mean(p) for p in probabilities]
        # print(summary)
        result = np.argmax(summary)
        if(not np.isnan(np.max(summary)) and np.max(summary) != np.min(summary)):
            result = result + 1
        return result
