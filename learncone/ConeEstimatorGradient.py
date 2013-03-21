# Bismillahi-r-Rahmani-r-Rahim

import logging

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

from ConeEstimatorBase import ConeEstimatorBase, positive

class ConeEstimatorGradient(ConeEstimatorBase):
    def decision_function(self, data):        
        logging.info("Predicting %d values", len(data))

        # Add a small constant to allow for rounding errors
        decisions = [min(np.dot(self.model,x)) + 1e-10 for x in data]
        logging.debug("First 100 decision values: %s", str(decisions[:100]))
        return decisions

    def learn_cone(self, vectors, class_values):
        orig_dims = len(vectors[0])
        estimate = random.random_sample(orig_dims*self.dimensions)*2 - 1
        estimate_shape = (self.dimensions, orig_dims) 
        estimate = estimate.reshape(estimate_shape)
        scale = None
        for i in range(300):
            logging.debug("Iteration %d", i)
            logging.debug("Estimate %s", str(estimate))
            difference, size = self.get_difference(vectors, class_values, estimate)
            logging.debug("Difference size: %f", size)
            if size == 0.0:
                break
            logging.debug("Difference %s", str(difference))

            if scale is None:
                scale = min(0.5, 10.0/size)
            logging.debug("Finding correct scale factor")    
            while True:
                new_estimate = estimate - scale*difference
                difference, new_size = self.get_difference(vectors, class_values, new_estimate)
                logging.debug("Size at scale factor %g: %f", scale, new_size)
                if new_size < size:
                    logging.debug("Found better size: %f", new_size)
                    break
                scale *= 0.5
                if scale < 1e-100:
                    logging.debug("Converged at iteration: %d", i)
                    self.model = estimate
                    return

            estimate = new_estimate
            scale *= 1.2
            logging.debug("Increasing scale to %g", scale)
        self.model = estimate

    def get_difference(self, vectors, class_values, estimate):
        mapped = np.dot(estimate, vectors.T)
        logging.debug("Mapped %s", str(mapped))
        fixed = self.project(mapped.T, class_values).T
        logging.debug("Fixed %s", str(fixed))
        difference = np.dot(mapped, vectors) - np.dot(fixed, vectors)
        size = np.sum(abs(difference))
        return difference, size
