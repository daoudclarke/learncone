# Bismillahi-r-Rahmani-r-Rahim
#
# Learn a cone by learning multiple intersecting planes


#import learnplane
import numpy as np
from numpy import random
from numpy.linalg import norm, inv
from math import sqrt

import svmlight as svm

from lattice import Lattice
import learnplane

np.random.seed(1)

def generate_basis(dimensions):
    basis = [random.random_sample(dimensions)*2 - 1.0 for i in range(dimensions)]
    normalised = [x/norm(x) for x in basis]
    return np.array(normalised).T

def sort_matrix(m):
    l = list(m.T)
    l.sort(lambda x,y: cmp(x[0], y[0]))
    return np.array(l).T

def basis_distance(b, c):
    b_sorted = sort_matrix(b)
    c_sorted = sort_matrix(c)
    difference = b_sorted - c_sorted
    return norm(difference)

def generate_data(lattice):
    docs = []
    class_values = []
    zero = np.zeros(lattice.dimensions)
    for i in xrange(400):
        v = random.random_sample(lattice.dimensions)*2 - 1.0
        doc = svm.Document(i, svm.SupportVector([
                    (j + 1, v[j]) for j in range(lattice.dimensions)]))
        docs.append(doc)
        pos = lattice.ge(v, zero)
        class_values.append(1 if pos else -1)
    return docs, class_values

def convert(docs, class_values):
    print class_values
    data = []
    for i in range(len(docs)):
        data.append(tuple(docs[i].vector) + (class_values[i] == 1.,) )
    return data

def learn_cone(docs, class_values, dimensions):
    planes = []
    for i in range(dimensions + 1):
        search = learnplane.cross_validate(docs, class_values)
        for params, mean_score, scores in search.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (
                mean_score, scores.std() / 2, params)
        print search.best_score_, search.best_params_

        plane = search.best_estimator_.model.plane
        learnplane.plot(convert(docs, class_values),
                        plane,
                        "iteration" + str(i),
                        [(-1.,1.),(-1.,1.)])
        planes.append(plane)
        # Remove negative class instances that the classifier gets right
        # i.e. true negatives 
        judgments = search.predict(docs)
        new_docs = []
        new_class_values = []
        for i in range(len(docs)):
            if not (class_values[i] == -1 and judgments[i] == -1):
                new_docs.append(docs[i])
                new_class_values.append(class_values[i])
        docs = new_docs
        class_values = new_class_values
    print planes
    #return np.array([x/norm(x) for x in planes[1:]])

    basis = inv(planes[1:])
    #basis = inv([-np.array(x) for x in planes[1:]])
    #print [list(x) for x in basis]
    basis = np.array([x/norm(x) for x in basis.T]).T
    return basis
        
     
def run():
    dimensions = 2
    basis = generate_basis(dimensions)
    #print basis
    lattice = Lattice(basis)
    #print lattice
    docs, class_values = generate_data(lattice)
    learnt = learn_cone(docs, class_values, dimensions)
    distance = basis_distance(basis, learnt)
    print "Original: "
    print basis
    print "Learnt: "
    print learnt
    print "Distance: ", distance 
    return distance

if __name__ == "__main__":
    runs = 10
    distances = [run() for i in range(runs)]
    print "Distances:"
    print distances
    d = np.array(distances)
    print "Mean:", d.mean(), "+/-", d.std()/sqrt(runs)
