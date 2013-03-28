# Bismillahi-r-Rahmani-r-Rahim

import logging

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

from ConeEstimatorBase import ConeEstimatorBase, positive
from ConeEstimatorGradient import ConeEstimatorGradient
from ConeEstimatorGreedy import ConeEstimatorGreedy

class ConeEstimatorKernel(ConeEstimatorGreedy):
    def decision_function(self, data):        
        logging.info("Predicting %d values", len(data))

        # Add a small constant to allow for rounding errors
        decisions = [min(np.dot(self.model,x)) + 1e-10 for x in data]
        logging.debug("First 100 decision values: %s", str(decisions[:100]))
        return decisions

    def learn_cone(self, vectors, class_values):
        K = np.dot(vectors, vectors.T)
        ConeEstimatorGreedy.learn_cone(self, K, class_values)
        self.model = np.dot(self.model, vectors)

