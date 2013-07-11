# Bismillahi-r-Rahmani-r-Rahim
#
# SVM estimator that matches recall specifications

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import safe_sparse_dot

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

from confusionmetrics.metrics import fbeta

class RecallSVMEstimator(LinearSVC):
    def __init__(self, beta=1.0, **params):
        super(RecallSVMEstimator, self).__init__(**params)
        self.beta = beta

    def get_params(self, deep=True):
        params = BaseEstimator.get_params(self, deep)
        params['beta'] = self.beta
        return params
    
    def set_params(self, **params):
        if 'beta' in params:
            self.beta = params['beta']
        return self

    def fit(self, data, class_values, **params):
        super(RecallSVMEstimator, self).fit(data, class_values, **params)
        positive_class = max(class_values)
        negative_class = min(class_values)
        are_positive = class_values == positive_class
        biases = safe_sparse_dot(data, self.coef_.T)
        bias_class = zip(biases, class_values)
        bias_class.sort()
        biases_sorted, classes_sorted = zip(*bias_class)
        classes_sorted = np.array(classes_sorted)
        #print "Positive Biases: ", biases
        #print "Metrics: ", metrics
        metrics = getMetrics(classes_sorted, positive_class, self.beta)
        optimal_index = np.argmax(metrics)
        if optimal_index == len(bias_class):
            self.intercept_ = -biases_sorted[optimal_index - 1] - 1e-10
        else:
            self.intercept_ = -biases_sorted[optimal_index] + 1e-10

def getMetrics(classes_sorted, positive_class, beta):
    metrics = []
    judgments = np.array([positive_class]*len(classes_sorted))
    confusion = confusion_matrix(classes_sorted, judgments)
    metrics.append(fbeta(confusion, beta))
    for i in range(len(classes_sorted)):
        if classes_sorted[i] == positive_class:
            confusion[1][1] -= 1
            confusion[1][0] += 1
        else:
            confusion[0][1] -= 1
            confusion[0][0] += 1
        metrics.append(fbeta(confusion, beta))
    return metrics
        
        # neg_biases = safe_sparse_dot(data[class_values != positive_class], self.coef_.T)
        # print "Negative biases: ", neg_biases

    # def predict(self, X):
    #     scores = self.decision_function(X)
    #     if len(scores.shape) == 1:
    #         indices = (scores > 0).astype(np.int)
    #     else:
    #         indices = scores.argmax(axis=1)
    #     return self.classes_[indices]

