# Bismillahi-r-Rahmani-r-Rahim

import unittest
import logging
from numpy import random

from learncone.RecallSVMEstimator import RecallSVMEstimator
from TestUtils import TestUtils

class RecallSVMEstimatorTestCase(unittest.TestCase, TestUtils):
    def setUp(self):
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)

    def testRecallSVMEstimatorArtificialData(self):
        classifier = RecallSVMEstimator(2)
        dataset = self.loadDataset('data/wn-noun-dependencies-10.mat')
        confusion = self.getConfusion(classifier, dataset)
        recall = float(confusion[1][1])/(confusion[1][0] + confusion[1][1])
        print "Confusion: %s, Recall: %f" % (str(confusion), recall)
        self.assertGreater(recall, 0.95)

