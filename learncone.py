# Bismillahi-r-Rahmani-r-Rahim
#
# Learn a cone by learning multiple intersecting planes


#import learnplane
import numpy as np
from numpy import random
from numpy.linalg import norm, inv, matrix_rank, det, pinv
from math import sqrt
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score
from scipy import optimize
from datetime import datetime

import logging

import csv
import svmlight as svm

from lattice import Lattice
import learnplane

np.random.seed(2)

def generate_basis(dimensions):
    basis = [random.random_sample(dimensions)*2 - 1.0 for i in range(dimensions)]
    normalised = [x/norm(x) for x in basis]
    if matrix_rank(normalised) != dimensions:
        raise ValueError("Singular matrix")
    print "Determinant:", det(normalised)
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
        # Generate positive data half the time
        if random.random_sample() > 0.5:
            v = random.random_sample(lattice.dimensions)
            v = np.array(np.dot(lattice.basis_matrix, v))[0]
        else:
            v = random.random_sample(lattice.dimensions)*2 - 1.0
        doc = svm.Document(i, svm.SupportVector([
                    (j + 1, v[j]) for j in range(lattice.dimensions)]))
        docs.append(doc)
        pos = lattice.ge(v, zero)
        class_values.append(1 if pos else -1)
    return docs, class_values

def convert(docs, class_values):
    #print class_values
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

def positive(m, v):
    product = np.dot(m, v)
    return (product >= -1e-10).all()

def add_if_safe(vectors, vector_to_add):
    """Only add the vector to the matrix if the new set of vectors
    are linearly independent."""
    if matrix_rank(vectors + [vector_to_add]) > len(vectors):
        vectors.append(vector_to_add)
    else:
        logging.debug("Skipping vector due to linear dependence")


# def get_initial(vectors, class_values, final_dimensions):
#     # Take the first d positive vectors and get their inverse
#     logging.info("Creating initial cone in %d dimensions from %d positive vectors",
#                  len(vectors[0]),
#                  len([x for x in class_values if x == 1.0]))
#     logging.debug("Class values: %s", str(class_values))
#     positives = []
#     i = 0
#     dimensions = len(vectors[0])
#     while len(positives) < dimensions:
#         if i == len(vectors):
#             logging.info("Adding %d random vectors to initial cone", dimensions - len(positives))
#         if i >= len(vectors):
#             add_if_safe(positives, random.random_sample(dimensions)*2 - 1.0) 
#         elif class_values[i] == 1.0:
#             add_if_safe(positives, vectors[i])
#         i += 1
#     initial = inv(np.array(positives).T)
#     normalised = np.array([x/norm(x) for x in initial])
#     # prod = np.dot(normalised, positives[0])
#     # print prod
#     # print norm(prod)
#     assert positive(normalised, positives[0])
#     reshaped = normalised.reshape(dimensions*dimensions)
#     return np.array([reshaped[i] for i in range(final_dimensions)])

def get_initial(vectors, class_values, dimensions):
    # Take the first d positive vectors and get their inverse
    logging.info("Creating initial cone in %d dimensions from %d positive vectors",
                 dimensions,
                 len([x for x in class_values if x == 1.0]))
    logging.debug("Class values: %s", str(class_values))
    positives = []
    i = 0
    while len(positives) < dimensions:
        if i >= len(vectors):
            raise ValueError("Not enough positive vectors for specified dimensions")
        if class_values[i] == 1.0:
            positives.append(vectors[i])
        i += 1
    inverse = pinv(positives)
    logging.debug("Inverse matrix: %s", str(inverse))
    return inverse.T.reshape(len(vectors[0])*dimensions)

def learn_cone_anneal(docs, class_values, dimensions):
    vectors = [np.array([x[1] for x in doc.vector])
                   for doc in docs]
    return learn_cone_anneal_vectors(
        vectors, class_values, dimensions)

def learn_cone_anneal_vectors(vectors, class_values, dimensions):
    #initial = random.random_sample(dimensions**2)*2 - 1.0
    initial = get_initial(vectors, class_values)
    #print initial
    
    upper = 1. #np.array([1.]*(dimensions**2))
    lower = -1. #np.array([-1.]*(dimensions**2))

    def fitness(vals):
        # Return a high score if out of range
        max_v = max(vals)
        min_v = min(vals)
        if max_v > 1.0 or min_v < -1.0:
            #print vals
            return max(max_v, -min_v)
        #print vals
        matrix = vals.reshape( (dimensions, dimensions) )
        #lattice = Lattice(matrix)
        #zero = np.zeros(lattice.dimensions)
        truth_map = {True:1, False:-1}
        predictions = [truth_map[positive(matrix, v)]
                       for v in vectors]
        return -f1_score(class_values, predictions)

    print "Initial fitness: ", fitness(initial)
    result = optimize.anneal(fitness, initial,
                             schedule = 'fast',
                             lower=-1,
                             upper=1,
                             T0=1e-12,
                             Tf=None,
                             maxaccept=2*dimensions,
                             maxiter=dimensions**2,
                             quench=1,
                             #dwell=dimensions**2,
                             feps=0.0,
                             full_output=True)
    #result = optimize.fmin(fitness, initial)

    print result
    learnt = result[0].reshape( (dimensions, dimensions) )
    basis = inv(learnt)
    return np.array([x/norm(x) for x in basis.T]).T

def learn_cone_descent(docs, class_values, dimensions):
    vectors = [np.array([x[1] for x in doc.vector])
                   for doc in docs]
    learnt = learn_cone_descent_vectors(docs, class_values, dimensions)
    basis = inv(learnt)
    return np.array([x/norm(x) for x in basis.T]).T

def learn_cone_descent_vectors(vectors, class_values, dimensions):
    initial = get_initial(vectors, class_values, dimensions)
    def fitness(vals):
        matrix = vals.reshape( (dimensions, len(vectors[0])) )
        truth_map = {True:1, False:0}
        predictions = [truth_map[positive(matrix, v)]
                       for v in vectors]
        logging.debug("Fitness precision: %f recall: %f",
                      precision_score(class_values, predictions),
                      recall_score(class_values, predictions))
        return f1_score(class_values, predictions)

    best = initial
    best_fitness = fitness(initial)
    logging.debug("Initial fitness: %f", best_fitness)
    accepts = 0
    for i in xrange(400):
        #diff = 0.01*(random.random_sample(dimensions**2)*2. - 1.)
        new = best.copy()
        diff = (random.random_sample()*2. - 1.)
        new[random.randint(dimensions*len(vectors[0]))] += diff
        new_fitness = fitness(new) 
        if new_fitness > best_fitness:
            logging.debug("Found new best fitness: %f", new_fitness)
            accepts += 1
            best = new
            best_fitness = new_fitness
        if (accepts > 5*dimensions**2
            or best_fitness > 0.95):
            break

    logging.info("Best fitness: %f", best_fitness)
    logging.info("Iterations: %d", i)
    learnt = best.reshape( (dimensions, len(vectors[0])) )
    return learnt
    
def get_stats(cone, docs, class_values):
    vectors = [np.array([x[1] for x in doc.vector])
                   for doc in docs]
    try:
        lattice = Lattice(cone)
    except:
        print "Unable to compute inverse for stats"
        return (0., 0., 0.)
    zero = np.zeros(lattice.dimensions)
    truth_map = {True:1, False:-1}
    predictions = [truth_map[lattice.ge(v, zero)]
                   for v in vectors]
    f1 = f1_score(class_values, predictions)
    precision = precision_score(class_values, predictions)
    recall = recall_score(class_values, predictions)
    return precision, recall, f1    
     
def run(dimensions, method, basis, docs, class_values):
    num_pos = len([x for x in class_values if x == 1.0])
    print "Total number of instances:", len(docs)
    print "Number of positives: ", num_pos
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
    #print "Original: "
    #print basis
    #print "Learnt: "
    #print learnt
    print "Distance: ", distance 
    print "Time: ", time
    return (distance,) + stats + (time,)

if __name__ == "__main__":
    runs = 10
    methods = {
        #"svm": learn_cone}
        #"anneal": learn_cone_anneal,
        "descent": learn_cone_descent,
        #"random": learn_cone_random
        }
    with open('results.csv', 'wb') as csvfile:
        results_file = csv.writer(csvfile)
        results_file.writerow(["Dimensions",
                               "Method",
                               "Distance", "Error",
                               "Precision", "Error",
                               "Recall", "Error",
                               "F1", "Error",
                               "Time", "Error"])
        for dimensions in [100]:
            cones = [generate_basis(dimensions) for i in range(runs)]
            lattices = [Lattice(cone) for cone in cones]
            data = [generate_data(lattice) for lattice in lattices]
            for method in methods.keys():
                results = [run(dimensions, methods[method], cones[i],
                               data[i][0], data[i][1]) for i in range(runs)]
                #print "Results"
                #print results
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
