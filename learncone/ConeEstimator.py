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
        self.model = self.learn_cone_descent_vectors(
            data, class_values)
        predictions = self.predict(data)
        logging.info("Training set precision: %f recall: %f f1: %f",
                      precision_score(class_values, predictions),
                      recall_score(class_values, predictions),
                      f1_score(class_values, predictions))
    
    def predict(self, data):
        return [1 if x > 0 else 0
                for x in self.decision_function(data)]
        #return [1 for x in data]
        # return [1 if self.model.classify(d) > 0 else -1
        #         for d in data]

    def decision_function(self, data):        
        logging.info("Predicting %d values", len(data))
        # Add a small constant to allow for rounding errors
        decisions = [min(np.dot(self.model, x)) + 1e-10 for x in data]
        logging.debug("First 100 decision values: %s", str(decisions[:100]))
        return decisions

    def get_initial(self, vectors, class_values):
        # Take the first d positive vectors and get their inverse
        logging.info("Creating initial cone in %d dimensions", self.dimensions)
        logging.debug("Using %d positive vectors",
                      len([x for x in class_values if x == 1.0]))
        logging.debug("Class values: %s", str(class_values))
        positives = []
        i = 0
        while len(positives) < self.dimensions:
            logging.debug("Iteration %d", i)
            if i >= len(vectors):
                raise ValueError("Not enough positive vectors for specified dimensions")
            if class_values[i] == 1.0:
                positives.append(vectors[i])
            i += 1
        logging.debug("Positives: %s", str(positives))
        inverse = pinv(positives)
        logging.debug("Inverse matrix: %s", str(inverse))
        return inverse.T.reshape(len(vectors[0])*self.dimensions)

    def learn_cone_descent_vectors(self, vectors, class_values):
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

        best = initial
        best_fitness = fitness(initial)
        logging.debug("Initial fitness: %f", best_fitness)
        accepts = 0
        for i in xrange(1000):
            #diff = 0.01*(random.random_sample(dimensions**2)*2. - 1.)
            new = best.copy()
            diff = (random.random_sample()*2. - 1.)
            new[random.randint(self.dimensions*len(vectors[0]))] += diff
            new_fitness = fitness(new) 
            if new_fitness > best_fitness:
                logging.debug("Found new best fitness: %f", new_fitness)
                accepts += 1
                best = new
                best_fitness = new_fitness
            if (accepts > 5*self.dimensions**2
                or best_fitness > 0.95):
                break

        logging.info("Best fitness: %f", best_fitness)
        logging.info("Iterations: %d", i)
        learnt = best.reshape( (self.dimensions, len(vectors[0])) )
        return learnt

