# Bismillahi-r-Rahmani-r-Rahim
#
# Estimator built on multiple SVM classifications

import logging

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn import preprocessing
#from sklearn.svm import LinearSVC

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

from RecallSVMEstimator import RecallSVMEstimator

class ConeEstimatorSVM(BaseEstimator):
    def __init__(self, dimensions=1, beta=1.0):
        self.dimensions = dimensions
        self.beta = beta

    def get_params(self, deep=True):
        params = BaseEstimator.get_params(self, deep)
        params['dimensions'] = self.dimensions
        params['beta'] = self.beta
        return params
    
    def set_params(self, **params):
        if 'dimensions' in params:
            self.dimensions = params['dimensions']
        if 'beta' in params:
            self.beta = params['beta']
        return self

    def fit(self, data, class_values):
        logging.info("Starting learning from %d data points",
                     len(data))
        if len(set(class_values)) != 2:
            raise ValueError('Need exactly two class values.')

        self.svcs = []
        for i in range(self.dimensions):
            if len(set(class_values)) != 2:
                logging.info("Only one class value remains, terminating learning")
                break
            logging.info("Starting learning iteration %d, data points %d",
                         i, len(class_values))
            svc = RecallSVMEstimator(self.beta)
            svc.fit(data, class_values)
            self.svcs.append(svc)

            self.positive_class = max(class_values)
            judgments = svc.predict(data)
            positives = class_values == self.positive_class
            incorrect_negatives = ((class_values != self.positive_class)
                                   & (judgments != class_values))
            new_indices = positives | incorrect_negatives
            data = data[new_indices]
            class_values = class_values[new_indices]
            

    def predict(self, data):
        predictions = np.array([self.positive_class]*len(data))
        for svc in self.svcs:
            new_predictions = svc.predict(data)
            predictions = np.minimum(predictions, new_predictions)
        return predictions

        return self.svc.predict(data)

