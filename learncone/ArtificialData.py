# Bismillahi-r-Rahmani-r-Rahim

import unittest

import numpy as np
from numpy import random
from numpy.linalg import pinv


from learncone.ConeEstimatorBase import positive
from learncone.ConeModel import ConeModel

from datetime import datetime, timedelta


class ArtificialData:
    def __init__(self, data_dims, cone_dims, size=3000):
        self.data_dims = data_dims
        self.cone_dims = cone_dims
        self.size = size

    def generate(self, noise=0.0, epsilon=0.0):
        rand_array =  random.random_sample(self.data_dims*self.cone_dims)*2 - 1
        self.cone = rand_array.reshape( (self.cone_dims, self.data_dims) )
        self.data = []
        self.target = []
        cone_inv = pinv(self.cone)
        self.model = ConeModel(self.cone, epsilon)
        for i in xrange(self.size):
            # Generate positive data half the time
            if random.random_sample() > 0.5:
                v = random.random_sample(self.cone_dims)
                v = np.array(np.dot(cone_inv, v))
            else:
                v = random.random_sample(self.data_dims)*2 - 1.0
            if random.random_sample() < noise:
                self.data.append(-v)
            else:
                self.data.append(v)
            self.target.append(self.model.predict([v])[0])
        self.data = np.array(self.data)
        self.target = np.array(self.target)

def make_data(data_dims, cone_dims, size=3000, noise=0.0, epsilon=0.0):
    data = ArtificialData(data_dims, cone_dims, size)
    data.generate(noise, epsilon)
    return data
