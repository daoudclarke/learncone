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
    def __init__(self, dimensions=1, noise=0.0, epsilon=0.0):
        if dimensions < 1:
            raise ValueError("Need at least one dimension to fit data.")
        self.dimensions = dimensions
        self.noise = noise
        self.epsilon = epsilon
        logging.debug("Initialised to %d dimensions", self.dimensions)

    def get_params(self, deep=True):
        params = BaseEstimator.get_params(self, deep)
        params['dimensions'] = self.dimensions
        params['noise'] = self.noise
        params['epsilon'] = self.epsilon
        logging.debug("Getting params: %s", str(params))
        return params
    
    def set_params(self, **params):
        logging.debug("Setting params")
        if 'dimensions' in params:
            self.dimensions = params['dimensions']
        if 'noise' in params:
            self.noise = params['noise']
        if 'epsilon' in params:
            self.epsilon = params['epsilon']
        return self

    def fit(self, data, input_class_values):
        self.encoder = preprocessing.LabelEncoder()
        logging.debug("Input classes: %s", str(set(input_class_values)))
        class_values = self.encoder.fit_transform(input_class_values)
        logging.debug("Encoder classes: %s", str(self.encoder.classes_))
        if len(self.encoder.classes_) <= 1:
            raise ValueError("Need at least two class values.")
        if self.dimensions < 1:
            raise ValueError("Need at least one dimension to fit data.")
        logging.info("Starting cone learning from %d data points in %d dimensions",
                     len(data), self.dimensions)
        self.learn_cone(data, class_values)
        predictions = self.predict(data)
        self.confusion = confusion_matrix(
            input_class_values, predictions).tolist()
        logging.info("Training set confusion: %s", str(self.confusion))

    def predict(self, data):
        results = [1 if x > -self.epsilon else 0
                   for x in self.decision_function(data)]
        return self.encoder.inverse_transform(results)

    def project(self, vectors, class_values):
        working = np.copy(vectors)
        eps = np.zeros(self.dimensions) - self.epsilon
        for i in range(len(class_values)):
            if class_values[i] == 1:
                # Vector in working should be positive
                working[i] = np.maximum(eps, working[i])
            else:
                # Subtract from vector if necessary to make it non-positive
                m = min(working[i])
                if m > -self.epsilon - 0.1:
                    index = np.argmin(working[i])
                    working[i][index] = -self.epsilon - 0.1
                    #working[i] = working[i] - (m + 0.1)
        return working
