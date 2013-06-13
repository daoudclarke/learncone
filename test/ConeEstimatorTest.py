# Bismillahi-r-Rahmani-r-Rahim

import unittest

import numpy as np
from numpy import random
from numpy.linalg import pinv

from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata, load_svmlight_file

from learncone.ConeEstimatorGradient import ConeEstimatorGradient
from learncone.ConeEstimatorFactorise import ConeEstimatorFactorise
from learncone.ConeEstimatorGreedy import ConeEstimatorGreedy
from learncone.ConeEstimatorBase import positive
from learncone.ConeEstimatorKernel import ConeEstimatorKernel
from learncone.ConeEstimator import ConeEstimator
from learncone.ArtificialData import make_data

from datetime import datetime, timedelta

import logging
logging.basicConfig(filename='results/unittest.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

class SvmLightDataset:
    def __init__(self, data, target):
        self.data = data.todense()
        self.target = target

class ConeEstimatorTestCase(unittest.TestCase):
    def setUp(self):
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)
    
    def testConeEstimatorFactoriseArtificialData(self):
        result, time = self.runArtificial(10, 3, ConeEstimatorFactorise(3))
        self.assertGreater(min(result), 0.8)

    def testConeEstimatorGradientArtificialData(self):
        result, time = self.runArtificial(10, 3, ConeEstimatorGradient(3))
        self.assertGreater(min(result), 0.85)

    @unittest.skip("Slow")
    def testConeEstimatorGreedyArtificialData(self):
        result, time = self.runArtificial(10, 3, ConeEstimatorGreedy(3))
        self.assertGreater(min(result), 0.7)

    @unittest.skip("Unreliable test")
    def testConeEstimatorKernelArtificialData(self):
        result, time = self.runArtificial(10, 3, ConeEstimatorKernel(3), 50)
        self.assertGreater(min(result), 0.7)

    @unittest.skip("Slow")
    def testConeEstimatorKernelWordNet(self):
        classifier = ConeEstimatorKernel(3)
        dataset = SvmLightDataset(*load_svmlight_file(
                'data/wn-noun-dependencies-10.mat'))
        result, time = self.runDataset(classifier, dataset, 100)
        self.assertGreater(min(result), 0.2)

    def testConeEstimatorGradientNoisyData(self):
        result, time = self.runArtificial(10, 3, ConeEstimatorGradient(3, 0.2), 50,
                                    generator = self.generateNoisyTestData)
        self.assertGreater(min(result), 0.6)

    def testConeEstimatorGradientShiftedData(self):
        epsilon = 1.5
        result, time = self.runArtificial(10, 3, ConeEstimatorGradient(3, 0.0, epsilon), 500,
                                    generator = self.generateApproximateTestData(epsilon))
        print "Accuracy: ", result
        self.assertGreater(min(result), 0.9)

    def testConeEstimatorApproximateNoisyWordNet(self):
        classifier = ConeEstimatorGradient(3, 0.15, -100.0) #  250000.0)
        dataset = SvmLightDataset(*load_svmlight_file(
                'data/wn-noun-dependencies-10.mat'))
        result, time = self.runDataset(classifier, dataset, 100)
        print "Accuracy: ", result
        self.assertGreater(min(result), 0.6)

    def testConeEstimatorGradientNoisyWordNet(self):
        classifier = ConeEstimatorGradient(3, 0.2)
        dataset = SvmLightDataset(*load_svmlight_file(
                'data/wn-noun-dependencies-10.mat'))
        result, time = self.runDataset(classifier, dataset, 100)
        self.assertGreater(min(result), 0.6)

    def testConeEstimatorGradientNoisyWordNetHighDimensional(self):
        classifier = ConeEstimatorGradient(10, 0.2, 0.1)
        dataset = SvmLightDataset(*load_svmlight_file(
                'data/wn-noun-dependencies-10.mat'))
        method = ShuffleSplit(len(dataset.target), n_iterations = 1, train_size = 300)
        train_indices = next(iter(method))[0]
        train_data = np.asarray(dataset.data)[train_indices]
        train_target = dataset.target[train_indices]
        classifier.fit(train_data, train_target)
        # result, time = self.runDataset(classifier, dataset, 100)
        confusion = classifier.confusion
        accuracy = (confusion[0][0] + confusion[1][1])/float(np.sum(confusion))
        print "Accuracy: %f, Confusion: %s" % (accuracy,confusion)
        self.assertGreater(accuracy, 0.6)

    def testConeEstimatorNoisyWordNet(self):
        classifier = ConeEstimator(3, 0.2)
        dataset = SvmLightDataset(*load_svmlight_file(
                'data/wn-noun-dependencies-10.mat'))
        result, time = self.runDataset(classifier, dataset, 100)
        self.assertGreater(min(result), 0.6)

    def testConeEstimatorFactoriseMnistDataset(self):
        result, time = self.runMnistDataset(ConeEstimatorFactorise(3))
        self.assertGreater(result, 0.5)
        self.assertLess(time, timedelta(seconds=60))

    def testConeEstimatorGradientMnistDataset(self):
        result, time = self.runMnistDataset(ConeEstimatorGradient(3))
        self.assertGreater(result, 0.6)
        self.assertLess(time, timedelta(seconds=120))

    def testConeEstimatorUnusualClassValues(self):
        result, time = self.runArtificial(10, 3, ConeEstimator(3),
                                          generator = self.generateMappedTestData)
        self.assertGreater(min(result), 0.85)

    def testConeEstimatorArtificialData(self):
        result, time = self.runArtificial(10, 3, ConeEstimator(3))
        self.assertGreater(min(result), 0.85)

    def testConeEstimatorMultiClassValues(self):
        result, time = self.runArtificial(10, 3, ConeEstimator(3),
                                          generator = self.generateMultiClassTestData)
        self.assertGreater(min(result), 0.7)

    def testConeEstimatorConsistency(self):
        random.seed(1006)
        dataset = self.generateTestData(2, 2)        
        accuracies = []
        for i in range(5):
            results = self.runDataset(ConeEstimator(2, 0.0),
                                      dataset, 100)
            print results
            accuracies.append(results[0])
        self.assertGreater(min(accuracies), 0.95)

    def runMnistDataset(self, classifier):
        dataset = fetch_mldata('mnist-original')
        return self.runDatasetBinary(classifier, dataset, 500)

    def runDatasetBinary(self, classifier, dataset, train_size):
        binary_map = np.vectorize(lambda x : 1 if x == 1 else 0)
        dataset.target = binary_map(dataset.target)
        return self.runDataset(classifier, dataset, train_size)

    def runDataset(self, classifier, dataset, train_size):
        if dataset.target.shape[0] > 500:
            method = ShuffleSplit(len(dataset.target), n_iterations = 1, train_size = train_size, test_size = 500)
        else:
            method = ShuffleSplit(len(dataset.target), n_iterations = 1)
        start = datetime.now()
        positive = max(dataset.target)
        result = cross_val_score(
            classifier,
            dataset.data,
            dataset.target,
            cv = method,
            score_func = lambda x,y: f1_score(x,y, pos_label = positive))
        logging.info("Classifier: %s, dataset F1: %f", str(classifier), result)
        time = datetime.now() - start
        logging.info("Test %s took time: %s",self._testMethodName, str(time))
        return result, time

    def runArtificial(self, data_dims, cone_dims, classifier, train_size = 500,
                      generator = None):
        """Construct an artificial dataset and test we can learn it"""
        if generator is None:
            generator = self.generateTestData
        logging.info("Generating %d dimensional cone in %d dimensions", cone_dims, data_dims)
        dataset = generator(data_dims, cone_dims)
        logging.info("Generated %d test data instances", len(dataset.target))
        return self.runDataset(classifier, dataset, train_size)

    def generateTestData(self, data_dims, cone_dims, num_instances=1000):
        return make_data(data_dims, cone_dims, size=num_instances)

    def generateNoisyTestData(self, data_dims, cone_dims, num_instances=1000):
        return make_data(data_dims, cone_dims, size=num_instances, noise=0.1)

    def generateApproximateTestData(self, epsilon):
        def generator(data_dims, cone_dims, num_instances=1000):
            return make_data(data_dims, cone_dims, size=num_instances, epsilon=epsilon)
        return generator

    def generateMappedTestData(self, data_dims, cone_dims):
        dataset = self.generateTestData(data_dims, cone_dims)
        m = {0: -1, 1: 7}
        dataset.target = np.array([m[x] for x in dataset.target])
        return dataset

    def generateMultiClassTestData(self, data_dims, cone_dims):
        dataset = self.generateTestData(data_dims, cone_dims)
        m = {0: lambda: random.randint(-2,0), 1: lambda: 1}
        dataset.target = np.array([m[x]() for x in dataset.target])
        return dataset

    
if __name__ == '__main__':
    unittest.main()
