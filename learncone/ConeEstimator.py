# Bismillahi-r-Rahmani-r-Rahim

import logging

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

def positive(m, v):
    product = np.dot(m, v)
    return (product >= -1e-10).all()

class ConeEstimatorBase(BaseEstimator):
    def __init__(self, dimensions):
        if dimensions < 1:
            raise ValueError("Need at least one dimension to fit data.")
        self.dimensions = dimensions
        logging.debug("Initialised to %d dimensions", self.dimensions)

    def get_params(self, deep=True):
        params = BaseEstimator.get_params(self, deep)
        params['dimensions'] = self.dimensions
        logging.debug("Getting params: %s", str(params))
        return params
    
    def set_params(self, **params):
        logging.debug("Setting params")
        if 'dimensions' in params:
            self.dimensions = params['dimensions']
        return self

class ConeEstimator(ConeEstimatorBase):
    def __init__(self, dimensions):
        ConeEstimatorBase.__init__(self, dimensions)

    def fit(self, data, class_values):
        self.classifier = OneVsRestClassifier(
            ConeEstimatorTwoClass(self.dimensions))
        self.classifier.fit(data, class_values)

    def predict(self, data):
        return self.classifier.predict(data)

class ConeEstimatorTwoClass(ConeEstimatorBase):
    def fit(self, data, class_values):
        if self.dimensions < 1:
            raise ValueError("Need at least one dimension to fit data.")
        logging.info("Starting cone learning from %d data points", len(data))
        self.model = self.learn_cone_gradient(
            data, class_values)
        predictions = self.predict(data)
        logging.info("Training set precision: %f recall: %f f1: %f",
                      precision_score(class_values, predictions),
                      recall_score(class_values, predictions),
                      f1_score(class_values, predictions))
    
    def predict(self, data):
        return [1 if x > 0 else 0
                for x in self.decision_function(data)]

    def decision_function(self, data):        
        logging.info("Predicting %d values", len(data))
        # Add a small constant to allow for rounding errors
        decisions = [min(np.dot(self.model, x)) + 1e-10 for x in data]
        logging.debug("First 100 decision values: %s", str(decisions[:100]))
        return decisions

    def learn_cone_gradient(self, vectors, class_values):
        orig_dims = len(vectors[0])
        estimate =  random.random_sample(orig_dims*self.dimensions)*2 - 1
        estimate_shape = (self.dimensions, orig_dims) 
        estimate = estimate.reshape(estimate_shape)
        zero = np.zeros(self.dimensions)
        for i in range(100):
            logging.debug("Iteration %d", i)
            logging.debug("Estimate %s", str(estimate))
            mapped = np.dot(estimate, vectors.T)
            logging.debug("Mapped %s", str(mapped))
            working = np.copy(mapped).T
            for i in range(len(vectors)):
                if class_values[i] == 1:
                    # Vector in working should be positive
                    working[i] = np.maximum(zero, working[i])
                else:
                    # Subtract from vector if necessary to make it non-positive
                    m = min(working[i])
                    if m > -0.1:
                        working[i] = working[i] - (m + 0.1)
            fixed = working.T
            logging.debug("Fixed %s", str(fixed))
            difference = np.dot(mapped, vectors) - np.dot(fixed, vectors)
            size = np.sum(abs(difference))
            logging.debug("Difference size: %f", size)
            if size == 0.0:
                break
            logging.debug("Difference %s", str(difference))
            estimate = estimate - min(0.5, 25.0/size)*difference
        return estimate
