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

        working = np.copy(vectors)
        for i in range(len(small)):
            index = small[i][0]
            working[index] = projected[index]
        return working
