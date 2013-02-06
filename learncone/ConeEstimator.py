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
        self.model = self.learn_cone_factorise(
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

    def learn_cone_factorise(self, vectors, class_values):
        # Factorise vectors = V into V ~= WH, minimise ||W - VH||
        orig_dims = len(vectors[0])
        num_docs = len(vectors)
        V = vectors.T
        
        # NB These are POSITIVE
        W = random.random_sample(orig_dims*self.dimensions)*2 - 1
        W = W.reshape( (orig_dims, self.dimensions) )

        H = random.random_sample(self.dimensions*num_docs)*2 - 1
        H = H.reshape( (self.dimensions, num_docs) )

        mu_w = 1
        mu_h = 1

        objectives = []
        for i in range(50):
            logging.debug("Iteration %d", i)
            new_obj = obj = np.linalg.norm(V - np.dot(W, H))
            
            # If the objective hasn't decreased much recently, stop
            if i > 3 and (sum(objectives)/len(objectives) - new_obj)/new_obj < 1e-5:
                logging.info("Objective stopped decreasing at: %f", new_obj)
                return W.T
            objectives.append(new_obj)
            objectives = objectives[-5:]
            while True:
                logging.debug("Objective: %f, mu_w: %f", new_obj, mu_w)
                W_new = W - mu_w*np.dot(np.dot(W,H) - V, H.T) 
                new_obj = np.linalg.norm(V - np.dot(W_new, H))
                if new_obj < obj:
                    break
                mu_w *= 0.5
                if mu_w <= 1e-200:
                    logging.info("Convergence at objective: %f", new_obj)
                    return W.T
            W = W_new
            mu_w *= 1.2
            logging.debug("Better W found at obj: %f", new_obj)

            obj = new_obj
            while True:
                logging.debug("Objective: %f, mu_h: %f", new_obj, mu_h)
                H_new = H - mu_h*np.dot(W.T, np.dot(W,H) - V)
                new_obj = np.linalg.norm(V - np.dot(W, H_new))
                if new_obj < obj:
                    break
                mu_h *= 0.5
                if mu_h <= 1e-200:
                    logging.info("Convergence at objective: %f", new_obj)
                    return W.T
            H = H_new
            mu_h *= 1.2
            logging.debug("Better H found at obj: %f", new_obj)
        return W.T
                   
        
        # for i in range(10):
        #     H = np.divide(np.multiply(
        #             H, np.dot(W.T, V),
        #             np.dot(W.T, np.dot(W, H))))
        #     logging.debug("H: %s", str(H))
        #     W = np.multiply(
        #         W, np.divide(np.dot(V, H.T),
        #                      np.dot(np.dot(W, H), H.T)))
        #     logging.debug("W: %s", str(W))
        #     size = np.linalg.norm(V - np.dot(W, H))
        #     logging.debug("Distance: %f", size)
        # return W.T

    def learn_cone_gradient(self, vectors, class_values):
        orig_dims = len(vectors[0])
        estimate = random.random_sample(orig_dims*self.dimensions)*2 - 1
        estimate_shape = (self.dimensions, orig_dims) 
        estimate = estimate.reshape(estimate_shape)
        for i in range(100):
            logging.debug("Iteration %d", i)
            logging.debug("Estimate %s", str(estimate))
            difference, size = self.get_difference(vectors, class_values, estimate)
            logging.debug("Difference size: %f", size)
            if size == 0.0:
                break
            logging.debug("Difference %s", str(difference))

            scale = min(0.5, 10.0/size)            
            estimate = estimate - scale*difference

            #scale = 0.5
            # new_size = size
            # new_estimate = estimate
            # while new_size >= size:
            #     new_estimate = estimate - scale*difference
            #     new_difference, new_size = self.get_difference(vectors, class_values, new_estimate)
            #     logging.debug("New estimate, scale: %f, new size %f", scale, new_size)
            #     scale = scale*0.5
            # estimate = new_estimate

            # if size < old_size:
            #     logging.debug("New best size")
            #     estimate = old_estimate - scale*difference
            #     scale = initial_scale
            #     old_size = size
            #     old_difference = difference
            #     old_estimate = estimate
            # else:
            #     scale *= 0.8
            #     logging.debug("New scale: %f", scale)
            #     estimate = old_estimate - scale*old_difference
        return estimate

    def get_difference(self, vectors, class_values, estimate):
        zero = np.zeros(self.dimensions)
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
                    index = np.argmin(working[i])
                    working[i][index] = -0.1
                    #working[i] = working[i] - (m + 0.1)
        fixed = working.T
        logging.debug("Fixed %s", str(fixed))
        difference = np.dot(mapped, vectors) - np.dot(fixed, vectors)
        size = np.sum(abs(difference))
        return difference, size
