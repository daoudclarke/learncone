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
    def __init__(self, max_dimensions=10, beta=1.0):
        self.max_dimensions = max_dimensions
        self.beta = beta

    def get_params(self, deep=True):
        params = BaseEstimator.get_params(self, deep)
        params['max_dimensions'] = self.max_dimensions
        params['beta'] = self.beta
        return params
    
    def set_params(self, **params):
        if 'max_dimensions' in params:
            self.max_dimensions = params['max_dimensions']
        if 'beta' in params:
            self.beta = params['beta']
        return self

    def fit(self, data, class_values):
        logging.info("Starting learning from %d data points",
                     len(data))
        if len(set(class_values)) != 2:
            raise ValueError('Need exactly two class values.')
        original_data = data
        original_class_values = class_values

        self.positive_class = max(class_values)
        self.svcs = []
        predictions = np.array([self.positive_class]*len(class_values))
        scores = []
        for i in range(self.max_dimensions):
            if len(set(class_values)) != 2:
                logging.info("Only one class value remains, terminating learning")
                break
            logging.info("Starting learning iteration %d, data points %d",
                         i, len(class_values))
            svc = RecallSVMEstimator(self.beta)
            svc.fit(data, class_values)
            self.svcs.append(svc)

            new_predictions = svc.predict(original_data)
            predictions = np.minimum(predictions, new_predictions)
            scores.append(f1_score(original_class_values, predictions))

            judgments = svc.predict(data)
            positives = class_values == self.positive_class
            incorrect_negatives = ((class_values != self.positive_class)
                                   & (judgments != class_values))
            new_indices = positives | incorrect_negatives
            data = data[new_indices]
            class_values = class_values[new_indices]
        logging.info("F1 scores for iterations: %s", str(scores))
        best_index = np.argmax(scores)
        self.svcs = self.svcs[:best_index + 1]
        logging.info("Number of SVM classifiers kept: %d", len(self.svcs))

    def predict(self, data):
        predictions = np.array([self.positive_class]*len(data))
        for svc in self.svcs:
            new_predictions = svc.predict(data)
            predictions = np.minimum(predictions, new_predictions)
        return predictions

        return self.svc.predict(data)

