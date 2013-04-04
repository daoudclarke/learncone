# Bismillahi-r-Rahmani-r-Rahim
#
# Wraps a cone estimator so it can deal with multiple classes

from sklearn.multiclass import OneVsRestClassifier

from ConeEstimatorBase import ConeEstimatorBase
from ConeEstimatorGradient import ConeEstimatorGradient

class ConeEstimator(ConeEstimatorBase):
    def fit(self, data, class_values):
        if len(set(class_values)) > 2:
            self.classifier = OneVsRestClassifier(
                ConeEstimatorGradient(self.dimensions))
            self.classifier.fit(data, class_values)
        else:
            self.classifier = ConeEstimatorGradient(
                self.dimensions)
            self.classifier.fit(data, class_values)


    def predict(self, data):
        return self.classifier.predict(data)
