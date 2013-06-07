# Bismillahi-r-Rahmani-r-Rahim
#
# Plot cones learnt in two dimensions

from numpy import random
from numpy.linalg import inv
import logging

from matplotlib import pyplot as pl
from matplotlib.patches import Polygon

from ArtificialData import ArtificialData
from ConeEstimator import ConeEstimator

def plot(dataset, cone, name, limits=[(-2.,2.),(-2.,2.)]):
    pl.clf()

    data = dataset.data
    class_values = dataset.target
    pos = [data[i] for i in range(len(class_values))
           if class_values[i] == 1]
    non_pos = [data[i] for i in range(len(class_values))
               if class_values[i] == 0]

    ax = pl.subplot(111)

    # print cone
    # rays = inv(cone)
    # for i in range(2):
    #     ray = cone[i]
    #     pl.plot([0,ray[0]], [0,ray[1]], linewidth=1)

    for i in range(2):
        plane = cone[i]
        gradient = -plane[0]/plane[1]

        def plane_func(x):
            return gradient*x

        pl.plot([0,1], [0,gradient], linewidth=1)

    # # make the shaded region
    # up = plane[1] > 0
    # verts = [(-2,-2*gradient),
    #          (-2 + plane[0], -2*gradient + plane[1]),
    #          (2 + plane[0], 2*gradient + plane[1]),
    #          (2, 2*gradient)]
    # poly = Polygon(verts, facecolor='0.8', edgecolor='k')
    # ax.add_patch(poly)
    # #pl.show()

    pl.plot([x[0] for x in pos], [x[1] for x in pos], 'ro')
    pl.plot([x[0] for x in non_pos], [x[1] for x in non_pos], 'bx')

    ax.set_xlim(left=limits[0][0], right=limits[0][1])
    ax.set_ylim(bottom=limits[1][0], top=limits[1][1])
    pl.savefig(name)

if __name__ == "__main__":
    logging.basicConfig(filename='../results/plot.log',
                        level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    #random.seed(1004)
    random.seed(1006)
    dataset = ArtificialData(2, 2, size=500)
    dataset.generate(epsilon=0.3)
    plot(dataset, dataset.cone, "../results/cone-true.png")
    estimator = ConeEstimator(2, 0.0, 0.3)
    estimator.fit(dataset.data, dataset.target)

    plot(dataset, estimator.classifier.model,"../results/cone.png")
    # for i in range(5):
    #     cone = estimator.classifier.estimates[i]
    #     plot(dataset, cone, "../results/cone%d.png" % i)
    #     #plot(dataset, estimator.classifier.projected[i], "../results/projected%d.png" % i)
