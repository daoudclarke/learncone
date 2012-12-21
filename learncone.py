# Bismillahi-r-Rahmani-r-Rahim

# Test of whether we can learn a cone using SVMs

import random

import numpy as np
import pylab as pl
from matplotlib.patches import Polygon
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score
from SvmLightEstimator import SvmLightEstimator

import svmlight as svm

def get_cone_data():
    """
    Get points in the unit square and classifications assuming
    a cone generated by (1,3) and (1,1)
    """
    data = []
    for i in range(1000):
        x = np.random.uniform()
        y = np.random.uniform()
        in_cone = (y/x) >= 1./3 and (y/x) <= 1.
        #in_cone = (y/x) <= 0.7
        data += [(x,y,in_cone)]
    return data

def score(y_true, y_pred, labels=None, pos_label=1, average='weighted'):
    return fbeta_score(y_true, y_pred, 2, labels, pos_label, average)

def cross_validate(data, class_values):
    pars = {'cost': [1.0, 10.0, 100.0],
            'cost_ratio': [float(x)/4 for x in range(3, 10)]}
    search = GridSearchCV(SvmLightEstimator(), pars, score_func = score)
    search.fit(data, class_values)
    return search

def get_docs_and_class_values(data):
    docs = []
    for i in range(len(data)):
        docs.append(svm.Document(i, svm.SupportVector(
                    [(1, data[i][0]), (2, data[i][1])])))
    m = {True: 1, False: -1}
    class_values = [m[x[2]] for x in data]
    return docs, class_values

def plot(data, plane):
    pos = [x for x in data if x[2]]
    neg = [x for x in data if not x[2]]

    gradient = -plane[0]/plane[1]

    def plane_func(x):
        return gradient*x

    ax = pl.subplot(111)

    #x = pl.arange(0, 1, 0.01)
    #y = plane_func(x)
    pl.plot([0,1], [0,gradient], linewidth=1)

    # make the shaded region
    up = plane[1] > 0
    #ix = pl.arange(a, b, 0.01)
    #iy = plane_func(ix)
    #verts = [(a,0)] + list(zip(ix,iy)) + [(b,0)]
    verts = [(0,0),
             (plane[0], plane[1]),
             (1 + plane[0], gradient + plane[1]),
             (1, gradient)]
    poly = Polygon(verts, facecolor='0.8', edgecolor='k')
    ax.add_patch(poly)
    #pl.show()

    pl.plot([x[0] for x in pos], [x[1] for x in pos], 'ro')
    pl.plot([x[0] for x in neg], [x[1] for x in neg], 'bx')

    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    pl.show()

if __name__ == "__main__":
    data = get_cone_data()
    docs, class_values = get_docs_and_class_values(data)
    #print docs, class_values
    
    search = cross_validate(docs, class_values)

    for params, mean_score, scores in search.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params)
   

    print search.best_score_, search.best_params_

    plane = search.best_estimator_.model.plane
    plot(data, plane)
