# Bismillahi-r-Rahmani-r-Rahim

import logging

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

def _positive(m, v):
    product = np.dot(m, v)
    return (product >= -1e-10).all()

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
        self.model = self.learn_cone_descent_vectors(
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

    def get_initial(self, vectors, class_values, dimensions):
        # Take the first d positive vectors and get their inverse
        logging.info("Creating initial cone in %d dimensions from %d positive vectors",
                     dimensions,
                     len([x for x in class_values if x == 1.0]))
        logging.debug("Class values: %s", str(class_values))
        positives = []
        i = 0
        while len(positives) < dimensions:
            if i >= len(vectors):
                raise ValueError("Not enough positive vectors for specified dimensions")
            if class_values[i] == 1.0:
                positives.append(vectors[i])
            i += 1
        inverse = pinv(positives)
        logging.debug("Inverse matrix: %s", str(inverse))
        return inverse.T.reshape(len(vectors[0])*dimensions)

    def learn_cone_descent_vectors(self, vectors, class_values, dimensions):
        initial = self.get_initial(vectors, class_values, dimensions)
        def fitness(vals):
            matrix = vals.reshape( (dimensions, len(vectors[0])) )
            truth_map = {True:1, False:0}
            predictions = [truth_map[_positive(matrix, v)]
                           for v in vectors]
            logging.debug("Fitness precision: %f recall: %f",
                          precision_score(class_values, predictions),
                          recall_score(class_values, predictions))
            return f1_score(class_values, predictions)

        best = initial
        best_fitness = fitness(initial)
        logging.debug("Initial fitness: %f", best_fitness)
        accepts = 0
        for i in xrange(400):
            #diff = 0.01*(random.random_sample(dimensions**2)*2. - 1.)
            new = best.copy()
            diff = (random.random_sample()*2. - 1.)
            new[random.randint(dimensions*len(vectors[0]))] += diff
            new_fitness = fitness(new) 
            if new_fitness > best_fitness:
                logging.debug("Found new best fitness: %f", new_fitness)
                accepts += 1
                best = new
                best_fitness = new_fitness
            if (accepts > 5*dimensions**2
                or best_fitness > 0.95):
                break

        logging.info("Best fitness: %f", best_fitness)
        logging.info("Iterations: %d", i)
        learnt = best.reshape( (dimensions, len(vectors[0])) )
        return learnt

