# Bismillahi-r-Rahmani-r-Rahim
#
# Implement a simple randomised greedy search for cone learning

import logging

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

from ConeEstimatorBase import ConeEstimatorBase, positive

class ConeEstimatorGreedy(ConeEstimatorBase):
    def decision_function(self, data):        
        logging.info("Predicting %d values", len(data))

        # Add a small constant to allow for rounding errors
        decisions = [min(np.dot(self.model,x)) + 1e-10 for x in data]
        logging.debug("First 100 decision values: %s", str(decisions[:100]))
        return decisions

    def learn_cone(self, vectors, class_values):
        initial = self.get_initial(vectors, class_values)
        def fitness(vals):
            matrix = vals.reshape( (self.dimensions, len(vectors[0])) )
            truth_map = {True:1, False:0}
            predictions = [truth_map[positive(matrix, v)]
                           for v in vectors]
            logging.debug("Fitness precision: %f recall: %f",
                          precision_score(class_values, predictions),
                          recall_score(class_values, predictions))
            return f1_score(class_values, predictions)

        num_entries = len(initial)
        best = initial
        best_fitness = fitness(initial)
        logging.debug("Initial fitness: %f", best_fitness)
        accepts = 0
        max_iters = 100*num_entries
        for i in xrange(max_iters):
            new = best.copy()
            diff = (random.random_sample()*2. - 1.)
            new[random.randint(self.dimensions*len(vectors[0]))] += diff
            new_fitness = fitness(new) 
            if new_fitness > best_fitness:
                logging.debug("Found new best fitness: %f", new_fitness)
                accepts += 1
                best = new
                best_fitness = new_fitness
            if (accepts >
                10*num_entries
                or best_fitness > 0.99):
                break

        if i == max_iters - 1:
            logging.warn("Reached maximum number of iterations without convergence")
        logging.info("Best fitness: %f after %d iterations and %d accepts",
                     best_fitness, i, accepts)
        learnt = best.reshape( (self.dimensions, len(vectors[0])) )
        self.model = learnt

    def get_initial(self, vectors, class_values):
        # Take the first d positive vectors and get their inverse
        logging.info("Creating initial cone in %d dimensions from %d positive vectors",
                     self.dimensions,
                     len([x for x in class_values if x == 1.0]))
        logging.debug("Class values: %s", str(class_values))
        positives = []
        i = 0
        while len(positives) < self.dimensions:
            if i >= len(vectors):
                raise ValueError("Not enough positive vectors for specified dimensions")
            if class_values[i] == 1.0:
                positives.append(vectors[i])
            i += 1
        inverse = pinv(positives)
        logging.debug("Inverse matrix: %s", str(inverse))
        return inverse.T.reshape(len(vectors[0])*self.dimensions)
