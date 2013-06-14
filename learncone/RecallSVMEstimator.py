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
        judgments = [positive_class]*len(class_values)
        metrics = []
        for i in range(len(bias_class)):
            metrics.append(fbeta_score(classes_sorted, judgments, self.beta))
            judgments[i] = negative_class
        #print classes_sorted
        metrics.append(fbeta_score(classes_sorted, judgments, self.beta))
        #print "Positive Biases: ", biases
        #print "Metrics: ", metrics
        optimal_index = np.argmax(metrics)
        if optimal_index == len(bias_class):
            self.intercept_ = -biases_sorted[optimal_index - 1] - 1e-10
        else:
            self.intercept_ = -biases_sorted[optimal_index] + 1e-10

        
        # neg_biases = safe_sparse_dot(data[class_values != positive_class], self.coef_.T)
        # print "Negative biases: ", neg_biases

    # def predict(self, X):
    #     scores = self.decision_function(X)
    #     if len(scores.shape) == 1:
    #         indices = (scores > 0).astype(np.int)
    #     else:
    #         indices = scores.argmax(axis=1)
    #     return self.classes_[indices]

