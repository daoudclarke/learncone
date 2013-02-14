# Bismillahi-r-Rahmani-r-Rahim

import unittest

import numpy as np
from numpy import random
from numpy.linalg import pinv

from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata

from learncone import ConeEstimator

from datetime import datetime, timedelta

import logging
logging.basicConfig(filename='results/unittest.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

class ConeEstimatorTestCase(unittest.TestCase):
    def setUp(self):
        random.seed(1001)
    
    def testArtificialDatasetTwoClass(self):
        data_dims = 10
        cone_dims = 3
        classifier = ConeEstimator.ConeEstimatorTwoClass(cone_dims)
        result = self.runClassifier(data_dims, cone_dims, classifier)
        self.assertGreater(result, 0.9)

    @unittest.skip("There is a problem with OneVsRest classifier "
                   + "that means it doesn't work with binary classes")
    def testArtificialDatasetMultiClass(self):
        data_dims = 10
        cone_dims = 3
        classifier = ConeEstimator.ConeEstimator(cone_dims)
        result = self.runClassifier(data_dims, cone_dims, classifier)
        print result

    def testMnistDataset(self):
        dataset = fetch_mldata('mnist-original')
        binary_map = np.vectorize(lambda x : 1 if x == 0 else 0)
        binary_target = binary_map(dataset.target)
        method = ShuffleSplit(len(dataset.target), n_iterations = 1, train_size = 300, test_size = 500)
        start = datetime.now()
        result = cross_val_score(
            ConeEstimator.ConeEstimatorTwoClass(3),
            dataset.data,
            binary_target,
            cv = method,
            score_func = f1_score)
        print "MNIST F1: ", result
        time = datetime.now() - start
        self.assertGreater(result, 0.6)
        self.assertLess(time, timedelta(seconds=3))

    def runClassifier(self, data_dims, cone_dims, classifier):
        """Construct an artificial dataset and test we can learn it"""
        rand_array =  random.random_sample(data_dims*cone_dims)*2 - 1
        cone = rand_array.reshape( (cone_dims, data_dims) )
        data, class_values = self.generateTestData(cone, data_dims, cone_dims)
        method = ShuffleSplit(len(data), n_iterations = 1, train_size = 100)
        result = cross_val_score(
            classifier, data,
            class_values,
            cv = method,
            score_func = f1_score)
        return result

    def generateTestData(self, cone, data_dims, cone_dims):
        data = []
        class_values = []
        cone_inv = pinv(cone)
        for i in xrange(3000):
            # Generate positive data half the time
            if random.random_sample() > 0.5:
                v = random.random_sample(cone_dims)
                v = np.array(np.dot(cone_inv, v))
            else:
                v = random.random_sample(data_dims)*2 - 1.0
            data.append(v)
            if ConeEstimator.positive(cone, v):
                class_values.append(1)
            else:
                class_values.append(0)
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
