# Bismillahi-r-Rahmani-r-Rahim
#
# Learn a cone by learning multiple intersecting planes


#import learnplane
import numpy as np
from numpy import random
from numpy.linalg import norm, inv
from math import sqrt
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score
from scipy import optimize
from datetime import datetime

import csv
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
    for i in xrange(2000):
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

def learn_cone_random(docs, class_values, dimensions):
    return generate_basis(dimensions)

def learn_cone(docs, class_values, dimensions):
    planes = []
    for i in range(dimensions + 1):
        try:
            search = learnplane.cross_validate(docs, class_values)
            for params, mean_score, scores in search.grid_scores_:
                print "%0.3f (+/-%0.03f) for %r" % (
                    mean_score, scores.std() / 2, params)
            print search.best_score_, search.best_params_

            plane = search.best_estimator_.model.plane
        except ValueError:
            # Not enough points
            plane = random.random_sample(dimensions)

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
    
    try:
        basis = inv(planes[1:])
    except np.linalg.linalg.LinAlgError:
        print "Unable to find independent planes"
        # Just use a random basis
        basis = generate_basis(dimensions)
    #basis = inv([-np.array(x) for x in planes[1:]])
    #print [list(x) for x in basis]
    basis = np.array([x/norm(x) for x in basis.T]).T
    return basis

def learn_cone_anneal(docs, class_values, dimensions):
    vectors = [np.array([x[1] for x in doc.vector])
                   for doc in docs]
    #print vectors

    initial = random.random_sample(dimensions**2)*2 - 1.0
    print initial
    
    upper = np.array([1.]*(dimensions**2))
    lower = np.array([-1.]*(dimensions**2))

    def fitness(vals):
        # Return a high score if out of range
        max_v = max(vals)
        min_v = min(vals)
        if max_v > 1.0 or min_v < -1.0:
            #print vals
            return max(max_v, -min_v)
        #print vals
        matrix = vals.reshape( (dimensions, dimensions) )
        lattice = Lattice(matrix)
        zero = np.zeros(lattice.dimensions)
        truth_map = {True:1, False:-1}
        predictions = [truth_map[lattice.ge(v, zero)]
                       for v in vectors]
        return -f1_score(class_values, predictions)

    print fitness(initial)
    result = optimize.anneal(fitness, initial,
                             schedule = 'fast',
                             lower=-1., #lower,
                             upper=1.,
                             quench=1) #upper)
    #result = optimize.fmin(fitness, initial)

    print result
    learnt = result[0].reshape( (dimensions, dimensions) )
    return np.array([x/norm(x) for x in learnt.T]).T
    
def get_stats(cone, docs, class_values):
    vectors = [np.array([x[1] for x in doc.vector])
                   for doc in docs]
    try:
        lattice = Lattice(cone)
    except:
        return (0., 0., 0.)
    zero = np.zeros(lattice.dimensions)
    truth_map = {True:1, False:-1}
    predictions = [truth_map[lattice.ge(v, zero)]
                   for v in vectors]
    f1 = f1_score(class_values, predictions)
    precision = precision_score(class_values, predictions)
    recall = recall_score(class_values, predictions)
    return precision, recall, f1    
     
def run(dimensions, method, basis):
    #print basis
    lattice = Lattice(basis)
    #print lattice
    docs, class_values = generate_data(lattice)
    train_size = len(docs)/2
    train_docs = docs[:train_size]
    train_class_values = class_values[:train_size]

    start = datetime.now()
    learnt = method(train_docs, train_class_values, dimensions)
    end = datetime.now()
    time = (end - start).total_seconds()
    distance = basis_distance(basis, learnt)

    test_docs = docs[train_size:]
    test_class_values = class_values[train_size:]
    stats = get_stats(learnt, test_docs, test_class_values)
    print "Original: "
    print basis
    print "Learnt: "
    print learnt
    print "Distance: ", distance 
    print "Time: ", time
    return (distance,) + stats + (time,)

if __name__ == "__main__":
    runs = 10
    methods = {"svm": learn_cone}
               #"anneal": learn_cone_anneal,
               #"random": learn_cone_random}
    with open('results.csv', 'wb') as csvfile:
        results_file = csv.writer(csvfile)
        results_file.writerow(["Dimensions",
                               "Method",
                               "Distance", "Error",
                               "Precision", "Error",
                               "Recall", "Error",
                               "F1", "Error",
                               "Time", "Error"])
        for dimensions in [4]:
            cones = [generate_basis(dimensions) for i in range(runs)]
            for method in methods.keys():
                results = [run(dimensions, methods[method], cone) for cone in cones]
                print "Results"
                print results
                distances = np.array([r[0] for r in results])
                print "Distance:", distances.mean(), "+/-", distances.std()/sqrt(runs)
                precisions = np.array([r[1] for r in results])
                print "Precision:", precisions.mean(), "+/-", precisions.std()/sqrt(runs)
                recalls = np.array([r[2] for r in results])
                print "Recall:", recalls.mean(), "+/-", recalls.std()/sqrt(runs)
                f1 = np.array([r[3] for r in results])
                print "F1:", f1.mean(), "+/-", f1.std()/sqrt(runs)
                times = np.array([r[4] for r in results])
                print "Time:", times.mean(), "+/-", times.std()/sqrt(runs)
                results_file.writerow([dimensions,
                                       method,
                                       distances.mean(), distances.std()/sqrt(runs),
                                       precisions.mean(), precisions.std()/sqrt(runs),
                                       recalls.mean(), recalls.std()/sqrt(runs),
                                       f1.mean(), f1.std()/sqrt(runs),
                                       times.mean(), times.std()/sqrt(runs)])
