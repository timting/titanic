#!/usr/bin/env python

from numpy import *
import operator
import titanic

def distances(data, point):
    point_matrix = tile(point, (data.shape[0],1))
    diffs = point_matrix - data
    square_diffs = diffs**2
    square_distances = square_diffs.sum(axis=1)
    dist = square_distances**0.5
    return dist

def nearest_neighbors(data, point, labels, k):
    d = distances(data, point)
    distance_indices = d.argsort()

    votes = {}
    for i in range(k):
        label = labels[ distance_indices[i] ]
        votes[label] = votes.get(label, 0) + 1

    sorted_labels = sorted(votes.iteritems(),
                           key=operator.itemgetter(1), reverse=True)

    return sorted_labels[0][0]

def main():
    data,labels,_ = titanic.read_data("normalized-train.csv")
    test,test_labels,_ = titanic.read_data("normalized-test.csv")

    successes = 0
    totals = 0
    for i in range(len(test)):
        point = test[i]

        knn_label = nearest_neighbors(data, point, labels, 3)
        # print " want %s, got %s" % (test_labels[i], knn_label)
        if knn_label == test_labels[i]:
            successes += 1
        totals += 1

    print "accuracy: %2.4f %%" % (float(successes) / float(totals) * 100.0)


if __name__ == "__main__":
    main()
