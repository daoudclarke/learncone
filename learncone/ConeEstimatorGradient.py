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
        #decisions = np.dot(self.model, data.T).min(axis=0).flatten() + 1e-10
        decisions = [min(np.dot(self.model,x)) + 1e-10 for x in data]
        logging.debug("First 100 decision values: %s", str(decisions[:100]))
        return decisions

    def learn_cone(self, vectors, class_values):
        estimates = []
        for i in range(10):
            self.initialise_estimate(vectors)
            self.learn_cone_iterate(vectors, class_values, 10)
            predictions = self.predict(vectors)
            score = f1_score(class_values, predictions)
            logging.info("Initial estimate %d: %f", i, score)
            estimates.append( (np.copy(self.model), score) )
        best_estimate = max(estimates, key = lambda x: x[1])
        logging.info("Using estimate with score %f", best_estimate[1])
        self.model = best_estimate[0]
        self.learn_cone_iterate(vectors, class_values, 300)

    def initialise_estimate(self, vectors):
        orig_dims = len(vectors[0])
        self.model = random.random_sample(orig_dims*self.dimensions)*2 - 1
        estimate_shape = (self.dimensions, orig_dims) 
        self.model = self.model.reshape(estimate_shape)

    def learn_cone_iterate(self, vectors, class_values, num_iterations):
        scale = None
        difference, size = self.get_difference(vectors, class_values, self.model)
        logging.debug("Initial difference size: %f", size)
        for i in range(num_iterations):
            logging.debug("Iteration %d", i)
            if size == 0.0:
                break

            if scale is None:
                scale = min(0.5, 10.0/size)
            logging.debug("Finding correct scale factor")    
            while True:
                new_estimate = self.model - scale*difference
                difference, new_size = self.get_difference(vectors, class_values, new_estimate)
                logging.debug("Size at scale factor %g: %f", scale, new_size)
                if new_size < size:
                    logging.debug("Found better size: %f", new_size)
                    break
                scale *= 0.5
                if scale < 1e-100:
                    logging.debug("Converged at iteration: %d", i)
                    return

            self.model = new_estimate
            size = new_size
            scale *= 1.2
            logging.debug("Increasing scale to %g", scale)
        self.model = self.model

    def get_difference(self, vectors, class_values, estimate):
        mapped = np.dot(estimate, vectors.T)
        fixed = self.project(mapped.T, class_values).T
        difference = np.dot(mapped - fixed, vectors)
        size = np.sum(abs(difference))
        return difference, size

    def get_small(self, differences):
        num_noisy = int(self.noise*len(differences))
        if num_noisy == 0:
            return differences
        ordered = sorted(differences, key=lambda x: x[1])
        # values = [x[1] for x in differences]
        # avg = sum(values)/len(values)
        logging.debug("Removing %d noisy instances", num_noisy)
        return ordered[:-num_noisy]
        
    def project(self, vectors, class_values):
        projected = ConeEstimatorBase.project(self, vectors, class_values)

        difference_sizes = np.sum(abs(vectors - projected), axis=1)
        differences = zip(xrange(len(difference_sizes)), difference_sizes)
        logging.debug("Total number of instances: %d", len(differences))

        pos = [differences[i] for i in range(len(class_values))
               if class_values[i] == 1]
        small = self.get_small(pos)
        logging.debug("Number of small positives: %d", len(small))

        non_pos = [differences[i] for i in range(len(class_values))
               if class_values[i] == 0]
        small += self.get_small(non_pos)
        logging.debug("Number of small pos and neg: %d", len(small))

        working = np.copy(vectors)
        for i in range(len(small)):
            index = small[i][0]
            working[index] = projected[index]
        return working
