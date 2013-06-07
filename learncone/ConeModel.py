# Bismillahi-r-Rahmani-r-Rahim

# Simple class to define a model

from ConeEstimatorGradient import ConeEstimatorGradient
from sklearn.base import BaseEstimator

class NullEncoder(BaseEstimator):
    def inverse_transform(self, results):
        return results

class ConeModel(ConeEstimatorGradient):
    def __init__(self, model, epsilon):
        self.model = model
        self.encoder = NullEncoder()
        super(ConeEstimatorGradient,self).__init__(epsilon = epsilon)

    
