# Bismillahi-r-Rahmani-r-Rahim
#
# Wraps a cone estimator so it can deal with multiple classes

from sklearn.multiclass import OneVsRestClassifier

from ConeEstimatorBase import ConeEstimatorBase
from ConeEstimatorGradient import ConeEstimatorGradient

class ConeEstimator(ConeEstimatorBase):
    def fit(self, data, class_values):
        if len(set(class_values)) > 2:
            subclassifier = ConeEstimatorGradient()
            subclassifier.set_params(**self.get_params())
            self.classifier = OneVsRestClassifier(subclassifier)
        else:
            self.classifier = ConeEstimatorGradient()
            self.classifier.set_params(**self.get_params())
        self.classifier.fit(data, class_values)


    def predict(self, data):
        return self.classifier.predict(data)
