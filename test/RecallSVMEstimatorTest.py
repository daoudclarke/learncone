# Bismillahi-r-Rahmani-r-Rahim

import unittest

from learncone.RecallSVMEstimator import RecallSVMEstimator
from TestUtils import TestUtils

class RecallSVMEstimatorTestCase(unittest.TestCase, TestUtils):
    def testRecallSVMEstimatorArtificialData(self):
        classifier = RecallSVMEstimator()
        dataset = self.loadDataset('data/wn-noun-dependencies-10.mat')
        confusion = self.getConfusion(classifier, dataset)
        recall = float(confusion[1][1])/(confusion[1][0] + confusion[1][1])
        print "Confusion: %s, Recall: %f" % (str(confusion), recall)
        self.assertGreater(recall, 0.95)

