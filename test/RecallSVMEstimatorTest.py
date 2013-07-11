# Bismillahi-r-Rahmani-r-Rahim

import unittest
import logging
from numpy import random

from sklearn.metrics import fbeta_score

from learncone.RecallSVMEstimator import RecallSVMEstimator, getMetrics
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

    def testGetMetrics(self):
        negative_class = 0
        positive_class = 1
        actual = random.random_integers(negative_class,positive_class,100)
        judgments = [positive_class]*len(actual)
        beta = 2.0
        expected_metrics = []
        for i in range(len(actual)):
            expected_metrics.append(fbeta_score(actual, judgments, beta))
            judgments[i] = negative_class
        expected_metrics.append(fbeta_score(actual, judgments, beta))    

        actual_metrics = getMetrics(actual, positive_class, beta)

        self.assertEqual(expected_metrics, actual_metrics)
