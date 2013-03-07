# Bismillahi-r-Rahmani-r-Rahim

import unittest

import numpy as np
from numpy import random
from numpy.linalg import pinv

from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata

from learncone.ConeEstimatorGradient import ConeEstimatorGradient
from learncone.ConeEstimatorFactorise import ConeEstimatorFactorise
from learncone.ConeEstimatorBase import positive
from learncone.ConeEstimator import ConeEstimator
from learncone.ArtificialData import make_data

from datetime import datetime, timedelta

import logging
logging.basicConfig(filename='results/unittest.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

class ConeEstimatorTestCase(unittest.TestCase):
    def setUp(self):
        logging.info("Starting test: %s", self._testMethodName)
        random.seed(1001)
    
    def testConeEstimatorFactoriseArtificialData(self):
        result = self.runArtificial(10, 3, ConeEstimatorFactorise(3))
        self.assertGreater(min(result), 0.9)

    def testConeEstimatorGradientArtificialData(self):
        result = self.runArtificial(10, 3, ConeEstimatorGradient(3))
        self.assertGreater(min(result), 0.9)

    def testConeEstimatorFactoriseMnistDataset(self):
        result, time = self.runMnistDataset(ConeEstimatorFactorise(3))
        self.assertGreater(result, 0.6)
        self.assertLess(time, timedelta(seconds=15))

    def testConeEstimatorGradientMnistDataset(self):
        result, time = self.runMnistDataset(ConeEstimatorGradient(3))
        self.assertGreater(result, 0.6)
        self.assertLess(time, timedelta(seconds=3))

    def testConeEstimatorUnusualClassValues(self):
        result = self.runArtificial(10, 3, ConeEstimatorFactorise(3),
                                    generator = self.generateMappedTestData)
        self.assertGreater(min(result), 0.9)

    def testConeEstimatorArtificialData(self):
        result = self.runArtificial(10, 3, ConeEstimator(3))
        self.assertGreater(min(result), 0.9)

    def testConeEstimatorMultiClassValues(self):
        result = self.runArtificial(10, 3, ConeEstimator(3),
                                    generator = self.generateMultiClassTestData)
        self.assertGreater(min(result), 0.7)

    def runMnistDataset(self, classifier):
        dataset = fetch_mldata('mnist-original')
        binary_map = np.vectorize(lambda x : 1 if x == 0 else 0)
        binary_target = binary_map(dataset.target)
        method = ShuffleSplit(len(dataset.target), n_iterations = 1, train_size = 300, test_size = 500)
        start = datetime.now()
        result = cross_val_score(
            classifier,
            dataset.data,
            binary_target,
            cv = method,
            score_func = f1_score)
        logging.info("Classifier: %s, MNIST dataset F1: %f", str(classifier), result)
        time = datetime.now() - start
        return result, time

    def runArtificial(self, data_dims, cone_dims, classifier,
                      generator = None):
        """Construct an artificial dataset and test we can learn it"""
        if generator is None:
            generator = self.generateTestData
        rand_array =  random.random_sample(data_dims*cone_dims)*2 - 1
        logging.info("Generating %d dimensional cone in %d dimensions", cone_dims, data_dims)
        cone = rand_array.reshape( (cone_dims, data_dims) )
        data, class_values = generator(cone, data_dims, cone_dims)
        logging.info("Generated %d test data instances", len(class_values))
        method = ShuffleSplit(len(class_values), n_iterations = 3, train_size = 500, test_size = 500)
        positive = max(class_values)
        result = cross_val_score(
            classifier, data,
            class_values,
            cv = method,
            score_func = lambda x,y: f1_score(x,y, pos_label = positive))
        return result

    def generateTestData(self, cone, data_dims, cone_dims):
        dataset = make_data(data_dims, cone_dims)
        return dataset.data, dataset.target

    def generateMappedTestData(self, cone, data_dims, cone_dims):
        data, class_values = self.generateTestData(cone, data_dims, cone_dims)
        m = {0: -1, 1: 7}
        class_values = [m[x] for x in class_values]
        return data, class_values

    def generateMultiClassTestData(self, cone, data_dims, cone_dims):
        data, class_values = self.generateTestData(cone, data_dims, cone_dims)
        m = {0: lambda: random.randint(-2,0), 1: lambda: 1}
        class_values = [m[x]() for x in class_values]
        return data, class_values

        # docs = []
        # class_values = []
        # zero = np.zeros(lattice.dimensions)
        # for i in xrange(2000):
        #     # Generate positive data half the time
        #     if random.random_sample() > 0.5:
        #         v = random.random_sample(lattice.dimensions)
        #         v = np.array(np.dot(lattice.basis_matrix, v))[0]
        #     else:
        #         v = random.random_sample(lattice.dimensions)*2 - 1.0
        #     doc = svm.Document(i, svm.SupportVector([
        #                 (j + 1, v[j]) for j in range(lattice.dimensions)]))
        #     docs.append(doc)
        #     pos = lattice.ge(v, zero)
        #     class_values.append(1 if pos else -1)
        # return docs, class_values

    # def testConstruction(self):
    #     self.assertEqual(str(Document(1,SupportVector([(1,1.),(2,1.),(3,1.)]))),
    #                      "Document(1, SupportVector({1: 1.0, 2: 1.0, 3: 1.0}))")

    # def testConstructionNeedsSupportVector(self):
    #     self.assertRaises(TypeError, Document, 2, [1,2,3])
    
if __name__ == '__main__':
    unittest.main()
