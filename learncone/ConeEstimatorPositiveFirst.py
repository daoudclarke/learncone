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

class ConeEstimatorPositiveFirst(ConeEstimatorGradient):
    # def decision_function(self, data):        
    #     logging.info("Predicting %d values", len(data))

    #     # Add a small constant to allow for rounding errors
    #     decisions = [min(np.dot(self.model,x)) + 1e-10 for x in data]
    #     logging.debug("First 100 decision values: %s", str(decisions[:100]))
    #     return decisions

    # def learn_cone(self, vectors, class_values):
    #     orig_dims = len(vectors[0])
    #     estimate = random.random_sample(orig_dims*self.dimensions)*2 - 1
    #     estimate_shape = (self.dimensions, orig_dims) 
    #     estimate = estimate.reshape(estimate_shape)
    #     scale = None
    #     for i in range(300):
    #         logging.debug("Iteration %d", i)
    #         logging.debug("Estimate %s", str(estimate))
    #         difference, size = self.get_difference(vectors, class_values, estimate)
    #         logging.debug("Difference size: %f", size)
    #         if size == 0.0:
    #             break
    #         logging.debug("Difference %s", str(difference))

    #         if scale is None:
    #             scale = min(0.5, 10.0/size)
    #         logging.debug("Finding correct scale factor")    
    #         while True:
    #             new_estimate = estimate - scale*difference
    #             difference, new_size = self.get_difference(vectors, class_values, new_estimate)
    #             logging.debug("Size at scale factor %g: %f", scale, new_size)
    #             if new_size < size:
    #                 logging.debug("Found better size: %f", new_size)
    #                 break
    #             scale *= 0.5
    #             if scale < 1e-100:
    #                 logging.debug("Converged at iteration: %d", i)
    #                 self.model = estimate
    #                 return

    #         estimate = new_estimate
    #         scale *= 1.2
    #         logging.debug("Increasing scale to %g", scale)
    #     self.model = estimate

    # def get_difference(self, vectors, class_values, estimate):
    #     mapped = np.dot(estimate, vectors.T)
    #     logging.debug("Mapped %s", str(mapped))
    #     fixed = self.project(mapped.T, class_values).T
    #     logging.debug("Fixed %s", str(fixed))
    #     difference = np.dot(mapped, vectors) - np.dot(fixed, vectors)
    #     size = np.sum(abs(difference))
    #     return difference, size

    def get_small(self, differences):
        values = [x[1] for x in differences]
        avg = sum(values)/len(values)
        return [x for x in differences if x[1] < avg]        

    def project(self, vectors, class_values):
        projected = ConeEstimatorBase.project(self, vectors, class_values)

        differences = []
        for i in range(len(projected)):
            differences.append( (i, np.sum(abs(vectors[i] - projected[i]))) )

        pos = [differences[i] for i in range(len(class_values))
               if class_values[i] == 1]
        small = self.get_small(pos)
        logging.debug("Number of small positives: %d", len(small))

        non_pos = [differences[i] for i in range(len(class_values))
               if class_values[i] == 0]
        small += self.get_small(non_pos)
        logging.debug("Number of positives: %d", len(small))

        # small = self.get_small(differences)

        working = np.copy(vectors)
        for i in range(len(small)):
            index = small[i][0]
            working[index] = projected[index]
        return working


    def project2(self, vectors, class_values):
        working = np.copy(vectors)
        zero = np.zeros(self.dimensions)
        # Only adjust elements that are close to being correct
        differences = {}
        for i in range(len(class_values)):
            if class_values[i] == 1:
                difference = np.sum(abs(np.maximum(zero, working[i]) - working[i]))
                differences[i] = difference

        avg = sum(differences.values())/len(differences)
        logging.debug("Average: %f, Differences: %s", avg, str(differences))

        for i in range(len(class_values)):
            if class_values[i] == 1 and differences[i] < avg:
                # Vector in working should be positive
                working[i] = np.maximum(zero, working[i])
            elif class_values[i] == 0:
                # Subtract from vector if necessary to make it non-positive
                m = min(working[i])
                if m > -0.1:
                    index = np.argmin(working[i])
                    working[i][index] = -0.1
                    #working[i] = working[i] - (m + 0.1)
        return working
