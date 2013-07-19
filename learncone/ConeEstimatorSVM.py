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
    def __init__(self, max_dimensions=10, beta=1.0, C=1.0):
        self.max_dimensions = max_dimensions
        self.beta = beta
        self.C = C

    def get_params(self, deep=True):
        params = BaseEstimator.get_params(self, deep)
        params['max_dimensions'] = self.max_dimensions
        params['beta'] = self.beta
        params['C'] = self.C
        return params
    
    def set_params(self, **params):
        if 'max_dimensions' in params:
            self.max_dimensions = params['max_dimensions']
        if 'beta' in params:
            self.beta = params['beta']
        if 'C' in params:
            self.C = params['C']
        return self

    def fit(self, data, original_class_values):
        logging.info("Starting learning from %d data points",
                     data.shape[0])
        if len(set(original_class_values)) != 2:
            raise ValueError('Need exactly two class values.')
        original_data = data
        self.original_class_values = original_class_values
        self.class_values = np.copy(original_class_values)

        self.positive_class = max(self.class_values)
        self.svcs = []
        self.predictions = np.array([self.positive_class]*len(self.class_values))
        self.scores = []
        for i in range(self.max_dimensions):
            if len(set(self.class_values)) != 2:
                logging.info("Only one class value remains, terminating learning")
                break
            logging.info("Starting learning iteration %d, data points %d",
                         i, len(self.class_values))
            svc = RecallSVMEstimator(self.beta, C=self.C)
            svc.fit(data, self.class_values)
            self.svcs.append(svc)

            new_indices = self.get_new_indices(original_data, data, svc)
            data = data[new_indices]
            self.class_values = self.class_values[new_indices]
        logging.info("F1 scores for iterations: %s", str(self.scores))
        best_index = np.argmax(self.scores)
        self.svcs = self.svcs[:best_index + 1]
        logging.info("Number of SVM classifiers kept: %d", len(self.svcs))

    def get_new_indices(self, original_data, data, svc):
        new_predictions = svc.predict(original_data)
        self.predictions = np.minimum(self.predictions, new_predictions)
        self.scores.append(f1_score(self.original_class_values, self.predictions))

        judgments = svc.predict(data)
        positives = self.class_values == self.positive_class
        incorrect_negatives = ((self.class_values != self.positive_class)
                               & (judgments != self.class_values))
        return np.nonzero(positives | incorrect_negatives)[0]

    def predict(self, data):
        predictions = np.array([self.positive_class]*data.shape[0])
        for svc in self.svcs:
            new_predictions = svc.predict(data)
            predictions = np.minimum(predictions, new_predictions)
        return predictions

        return self.svc.predict(data)

