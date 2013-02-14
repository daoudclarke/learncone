# Bismillahi-r-Rahmani-r-Rahim

from ConeEstimator import ConeEstimator

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
#from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.datasets import fetch_mldata
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score

import logging
import profile

def run():
    logging.info("Starting test")
    # cone_pipeline = Pipeline([('feature selection', SelectKBest(k=100)),
    #                           ('classification', ConeEstimator())])
    # cone_pipeline = Pipeline([('random PCA', RandomizedPCA(n_components=50)),
    #                           ('classification', ConeEstimator(3))])
    classifiers = [DecisionTreeClassifier(),
                   MultinomialNB(),
                   ConeEstimator(10)]
        #cone_pipeline]

    dataset = fetch_mldata('mnist-original')
    print "Dataset size: ", len(dataset.data)
    print "Features: ", len(dataset.data[0])

    for classifier in classifiers:
        method = ShuffleSplit(len(dataset.data), n_iterations = 1, train_size = 2000, test_size = 1000)
        result = cross_val_score(
            classifier, dataset.data,
            dataset.target,
            cv = method,
            score_func = f1_score)
        print classifier, result
    logging.info("Test complete")

if __name__ == "__main__":
    logging.basicConfig(filename='../results/mldatatest.log',
                        level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    try:
        #profile.run("run()")
        run()
    except:
        logging.exception("Exception on test run - aborting")
        raise
