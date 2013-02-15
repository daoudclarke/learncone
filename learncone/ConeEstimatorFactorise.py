# Bismillahi-r-Rahmani-r-Rahim

import logging

from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv

from ConeEstimatorBase import ConeEstimatorBase, positive

class ConeEstimatorFactorise(ConeEstimatorBase):
    def decision_function(self, data):        
        logging.info("Predicting %d values", len(data))
        H = self.get_decision_H(data)

        # Add a small constant to allow for rounding errors
        decisions = [min(x) + 1e-10 for x in H.T]
        logging.debug("First 100 decision values: %s", str(decisions[:100]))
        return decisions

    def get_decision_H(self, data):
        num_docs = len(data)
        V = data.T
        H = random.random_sample(self.dimensions*num_docs)*2 - 1
        H = H.reshape( (self.dimensions, num_docs) )
        mu_h = 1
        best_obj = float("inf")
        for i in range(500):
            logging.debug("Prediction iteration: %d", i)
            while True:
                H_new = H - mu_h*np.dot(self.W.T, np.dot(self.W,H) - V)
                new_obj = np.linalg.norm(V - np.dot(self.W, H_new))                
                logging.debug("Objective: %f, mu_h: %f", new_obj, mu_h)
                if new_obj < best_obj:
                    if (best_obj - new_obj)/new_obj < 1e-8:
                        logging.info("Objective stopped decreasing at: %f", new_obj)
                        return H_new
                    H = H_new
                    best_obj = new_obj
                    break
                mu_h *= 0.5
                if mu_h <= 1e-200:
                    logging.info("Convergence at objective: %f", new_obj)
                    return H
            mu_h *= 1.2

    def learn_cone(self, vectors, class_values):
        # Factorise vectors = V into V ~= WH, minimise ||W - VH||
        orig_dims = len(vectors[0])
        num_docs = len(vectors)
        V = vectors.T
        
        W = random.random_sample(orig_dims*self.dimensions)*2 - 1
        W = W.reshape( (orig_dims, self.dimensions) )

        H = random.random_sample(self.dimensions*num_docs)*2 - 1
        H = H.reshape( (self.dimensions, num_docs) )

        mu_w = 1
        mu_h = 1

        objectives = []
        for i in range(500):
            logging.debug("Iteration %d", i)
            new_obj = obj = np.linalg.norm(V - np.dot(W, H))
            logging.debug("Objective: %f", new_obj)
            
            # If the objective hasn't decreased much recently, stop
            if i > 3 and (sum(objectives)/len(objectives) - new_obj)/new_obj < 1e-5:
                logging.info("Objective stopped decreasing at: %f", new_obj)
                return
            objectives.append(new_obj)
            objectives = objectives[-5:]
            while True:
                logging.debug("Objective: %f, mu_w: %f", new_obj, mu_w)
                W_new = W - mu_w*np.dot(np.dot(W,H) - V, H.T) 
                new_obj = np.linalg.norm(V - np.dot(W_new, H))
                self.W = W_new
                if new_obj < obj:
                    break
                mu_w *= 0.5
                if mu_w <= 1e-200:
                    logging.info("Convergence at objective: %f", new_obj)
                    return
            W = W_new
            mu_w *= 1.2
            logging.debug("Better W found at obj: %f", new_obj)

            obj = new_obj
            while True:
                logging.debug("Objective: %f, mu_h: %f", new_obj, mu_h)
                H_new = H - mu_h*np.dot(W.T, np.dot(W,H) - V)
                # Make compatible with classifications
                #logging.debug("Before project: %s", str(H_new))
                H_new = self.project(H_new.T, class_values).T
                #logging.debug("After project: %s", str(H_new))
                new_obj = np.linalg.norm(V - np.dot(W, H_new))                
                if new_obj < obj:
                    break
                mu_h *= 0.5
                if mu_h <= 1e-200:
                    logging.info("Convergence at objective: %f", new_obj)
                    return
            H = H_new
            mu_h *= 1.2
            logging.debug("Better H found at obj: %f", new_obj)

