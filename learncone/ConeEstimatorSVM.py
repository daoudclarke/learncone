# Bismillahi-r-Rahmani-r-Rahim
#
# Estimator built on multiple SVM classifications

import logging

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.svm import LinearSVC

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

class ConeEstimatorSVM(BaseEstimator):
    def get_params(self, deep=True):
        params = BaseEstimator.get_params(self, deep)
        return params
    
    def set_params(self, **params):
        return self

    def fit(self, data, class_values):
        logging.info("Starting learning from %d data points",
                     len(data))

        self.svc = LinearSVC()
        self.svc.fit(data, class_values)

    def predict(self, data):
        return self.svc.predict(data)

