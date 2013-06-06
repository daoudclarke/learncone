# Bismillahi-r-Rahmani-r-Rahim

import unittest

import numpy as np
from numpy import random
from numpy.linalg import pinv


from learncone.ConeEstimatorBase import positive

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
        for i in xrange(self.size):
            # Generate positive data half the time
            is_positive = False
            if random.random_sample() > 0.5:
                v = random.random_sample(self.cone_dims)
                v += random.random_sample(self.cone_dims)*(2*epsilon) - epsilon
                v = np.array(np.dot(cone_inv, v))
                is_positive = True
            else:
                v = random.random_sample(self.data_dims)*2 - 1.0
                is_positive = positive(self.cone, v)
            if random.random_sample() < noise:
                self.data.append(-v)
            else:
                self.data.append(v)
            value = 1 if is_positive else 0
            self.target.append(value)
        self.data = np.array(self.data)
        self.target = np.array(self.target)

def make_data(data_dims, cone_dims, size=3000, noise=0.0):
    data = ArtificialData(data_dims, cone_dims, size)
    data.generate(noise)
    return data
