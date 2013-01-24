# Bismillahi-r-Rahmani-r-Rahim

import logging

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np

import learncone

class ConeEstimator(OneVsRestClassifier):
    def __init__(self):
        OneVsRestClassifier.__init__(
            self, ConeEstimatorTwoClass())

class ConeEstimatorTwoClass(BaseEstimator):
    def __init__(self):
        pass

    def get_params(self, deep=True):
        return {}
        # return {'cost': self.svm.cost,
        #         'cost_ratio': self.svm.cost_ratio,
        #         'biased_hyperplane': self.svm.biased_hyperplane}
    
    def set_params(self, **params):
        # if 'cost' in params:
        #     self.svm.cost = params['cost']
        # if 'cost_ratio' in params:
        #     self.svm.cost_ratio = params['cost_ratio']
        # if 'biased_hyperplane' in params:
        #     self.svm.biased_hyperplane = params['biased_hyperplane']
        return self

    def fit(self, data, class_values):
        logging.info("Starting cone learning from %d data points", len(data))
        self.model = learncone.learn_cone_descent_vectors(
            data, class_values, 5)
        predictions = self.predict(data)
        logging.info("Training set precision: %f recall: %f f1: %f",
                      precision_score(class_values, predictions),
                      recall_score(class_values, predictions),
                      f1_score(class_values, predictions))
    
    def predict(self, data):
        return [1 if x >= -1e-10 else 0
                for x in self.decision_function(data)]
        #return [1 for x in data]
        # return [1 if self.model.classify(d) > 0 else -1
        #         for d in data]

    def decision_function(self, data):        
        logging.info("Predicting %d values", len(data))
        decisions = [min(np.dot(self.model, x)) for x in data]
        logging.debug("First 100 decision values: %s", str(decisions[:100]))
        return decisions
