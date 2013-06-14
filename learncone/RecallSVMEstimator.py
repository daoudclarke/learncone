# Bismillahi-r-Rahmani-r-Rahim
#
# SVM estimator that matches recall specifications

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import safe_sparse_dot

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

class RecallSVMEstimator(LinearSVC):
    def fit(self, data, class_values, **params):
        super(RecallSVMEstimator, self).fit(data, class_values, **params)
        positive_class = max(class_values)
        are_positive = class_values == positive_class
        biases = safe_sparse_dot(data[are_positive], self.coef_.T)
        #print "Positive Biases: ", biases
        self.intercept_ = -biases.min() + 1e-10
        
        # neg_biases = safe_sparse_dot(data[class_values != positive_class], self.coef_.T)
        # print "Negative biases: ", neg_biases

    # def predict(self, X):
    #     scores = self.decision_function(X)
    #     if len(scores.shape) == 1:
    #         indices = (scores > 0).astype(np.int)
    #     else:
    #         indices = scores.argmax(axis=1)
    #     return self.classes_[indices]

