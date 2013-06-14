# Bismillahi-r-Rahmani-r-Rahim

import unittest

import numpy as np
from numpy import random
from numpy.linalg import pinv

from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata, load_svmlight_file

from datetime import datetime, timedelta

import logging
logging.basicConfig(filename='results/unittest.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

class SvmLightDataset:
    def __init__(self, data, target):
        self.data = data.todense()
        self.target = target

class TestUtils():
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

    def loadDataset(self, name):
        return SvmLightDataset(*load_svmlight_file(name))

    def getConfusion(self, classifier, dataset):
        method = ShuffleSplit(len(dataset.target), n_iterations = 1, train_size = 300)
        train_indices = next(iter(method))[0]
        train_data = np.asarray(dataset.data)[train_indices]
        train_target = dataset.target[train_indices]
        classifier.fit(train_data, train_target)
        predictions = classifier.predict(train_data)
        return confusion_matrix(train_target, predictions).tolist()
    
if __name__ == '__main__':
    unittest.main()
