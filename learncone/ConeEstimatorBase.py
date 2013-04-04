# Bismillahi-r-Rahmani-r-Rahim

import logging

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn import preprocessing

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

def positive(m, v):
    product = np.dot(m, v)
    return (product >= -1e-10).all()

class ConeEstimatorBase(BaseEstimator):
    def __init__(self, dimensions=1, noise=0.0):
        if dimensions < 1:
            raise ValueError("Need at least one dimension to fit data.")
        self.dimensions = dimensions
        self.noise = noise
        logging.debug("Initialised to %d dimensions", self.dimensions)

    def get_params(self, deep=True):
        params = BaseEstimator.get_params(self, deep)
        params['dimensions'] = self.dimensions
        params['noise'] = self.noise
        logging.debug("Getting params: %s", str(params))
        return params
    
    def set_params(self, **params):
        logging.debug("Setting params")
        if 'dimensions' in params:
            self.dimensions = params['dimensions']
        if 'noise' in params:
            self.noise = params['noise']
        return self

    def fit(self, data, input_class_values):
        self.encoder = preprocessing.LabelEncoder()
        class_values = self.encoder.fit_transform(input_class_values)
        if self.dimensions < 1:
            raise ValueError("Need at least one dimension to fit data.")
        logging.info("Starting cone learning from %d data points in %d dimensions",
                     len(data), self.dimensions)
        self.learn_cone(data, class_values)
        predictions = self.predict(data)
        self.confusion = confusion_matrix(
            input_class_values, predictions).tolist()
        logging.debug("Training set confusion: %s", str(self.confusion))

    def predict(self, data):
        results = [1 if x > 0 else 0
                   for x in self.decision_function(data)]
        return self.encoder.inverse_transform(results)

    def project(self, vectors, class_values):
        working = np.copy(vectors)
        zero = np.zeros(self.dimensions)
        for i in range(len(class_values)):
            if class_values[i] == 1:
                # Vector in working should be positive
                working[i] = np.maximum(zero, working[i])
            else:
                # Subtract from vector if necessary to make it non-positive
                m = min(working[i])
                if m > -0.1:
                    index = np.argmin(working[i])
                    working[i][index] = -0.1
                    #working[i] = working[i] - (m + 0.1)
        return working
