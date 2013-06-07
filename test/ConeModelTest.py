import unittest
import logging
import numpy as np
from numpy import random

from learncone.ConeModel import ConeModel

class ConeModelTestCase(unittest.TestCase):
    def setUp(self):
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)
    
    def testConeModelPredictNegativeEpsilon(self):
        matrix = np.array([[1,0],[0,1]])
        model = ConeModel(matrix, -0.5)
        self.assertEqual(model.predict([[0,0],
                                        [0,1],
                                        [1,0],
                                        [1,1]]),
                         [0,0,0,1])

    def testConeModelPredictPositiveEpsilon(self):
        matrix = np.array([[1,0],[0,1]])
        model = ConeModel(matrix, 0.5)
        self.assertEqual(model.predict([[0,0],
                                        [0,1],
                                        [1,0],
                                        [1,1]]),
                         [1,1,1,1])
