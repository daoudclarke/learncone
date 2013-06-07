import unittest
import logging
import numpy as np
from numpy import random

from learncone.ConeEstimatorBase import ConeEstimatorBase

class ConeModelTestCase(unittest.TestCase):
    def setUp(self):
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)

    def assertAlmostEqual(self, a, b, delta=1e-5):
        if not (((abs(a - b)) < delta).all()):
            raise AssertionError("Vectors are not equal: %s, %s" % (a,b))

    def testProjectionPositiveFromNegative(self):
        cone = ConeEstimatorBase(2)
        projected = cone.project([[1,-1]],[1])
        self.assertAlmostEqual(projected, [1,0])

    def testProjectionPositiveFromPositive(self):
        cone = ConeEstimatorBase(2)
        projected = cone.project([[1.,1.]],[1])
        self.assertAlmostEqual(projected, [1.,1.])

    def testProjectionNegativeFromPositive(self):
        cone = ConeEstimatorBase(2)
        projected = cone.project([[1.,0.5]],[0])
        self.assertAlmostEqual(projected, [1.,-0.1])

    def testProjectionNegativeFromNegative(self):
        cone = ConeEstimatorBase(2)
        projected = cone.project([[1.,-0.5]],[0])
        self.assertAlmostEqual(projected, [1.,-0.5])

    def testProjectionNegativeFromPositiveEpsilon(self):
        cone = ConeEstimatorBase(2, epsilon=0.5)
        projected = cone.project([[1.,0.9]],[0])
        self.assertAlmostEqual(projected, [1,-0.6])

    def testProjectionPositiveFromNegativeEpsilon(self):
        cone = ConeEstimatorBase(2, epsilon=0.5)
        projected = cone.project([[1.,-0.9]],[1])
        self.assertAlmostEqual(projected, [1,-0.5])
