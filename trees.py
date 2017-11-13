#!/usr/bin/env python

from numpy import *
import operator
import titanic
from math import log


def shannon_entropy(labels):
    num_entries = labels.shape[0]
    label_counts = {}
    for i in range(num_entries):
        label = labels[i]
        label_counts[label] = label_counts.get(label, 0) + 1
    entropy = 0.0
    for label in label_counts:
        probability = float(label_counts[label])/num_entries
        entropy -= probability * log(probability, 2)
    return entropy

def split_data(data, column_num, value):
    split = array([])
    for row in data:
        if row[column_num] == value:
            new_row = append(row[:column_num], row[(column_num+1):])
            if split.shape[0] == 0:
                split = array([new_row])
            else:
                split = vstack([split, new_row])
    return split

def choose_feature_to_split(data_set):
    num_features = data_set.shape[1] - 1
    base_entropy = shannon_entropy(data_set[:,-1])

    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        features = data_set[:,i]
        unique_features = set(features)
        new_entropy = 0.0
        for value in unique_features:
            value_dataset = split_data(data_set, i, value)
            prob = len(value_dataset) / float(len(data_set))
            new_entropy += prob * shannon_entropy(value_dataset[:,-1])
        info_gain = base_entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_vote(labels):
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    sorted_labels = sorted(label_counts.iteritems(),
                           key=operator.itemgetter(1),
                           reverse=True)
    return sorted_labels[0][0]

def data_set(data, labels):
    return hstack((data, array([labels]).T))

def create_tree(data_set, feature_names):
    labels = data_set[:,-1]
    if len(unique(labels)) == 1:
        return labels[0]
    if data_set.shape[1] == 1:
        return majority_vote(labels)
    best_feature = choose_feature_to_split(data_set)
    best_feature_name = feature_names[best_feature]

    tree = { best_feature_name: {} }
    feature_names_copy = feature_names[:]
    del(feature_names_copy[best_feature])

    feature_values = data_set[:,best_feature]
    unique_values = set(feature_values)

    for value in unique_values:
        tree[best_feature_name][value] = create_tree(split_data(data_set, best_feature, value), feature_names_copy)

    return tree

def print_tree(tree, depth=0):
    feature_name = tree.keys()[0]
    sub_tree = tree[feature_name]
    for key in sub_tree.keys():
        if type(sub_tree[key]).__name__ == 'dict':
            print "%s%s = %s" % ("  "*depth, feature_name, key)
            print_tree(sub_tree[key], depth+1)
        else:
            print "%s%s = %s => %s" % ("  "*depth, feature_name, key, sub_tree[key])

def tree_classify(tree, feature_names, input):
    first_feature_name = tree.keys()[0]
    sub_tree = tree[first_feature_name]
    feature_index = feature_names.index(first_feature_name)
    class_label = -1
    for key in sub_tree.keys():
        if input[feature_index] == float(key):
            if type(sub_tree[key]).__name__ == 'dict':
                class_label = tree_classify(sub_tree[key], feature_names, input)
            else:
                class_label = sub_tree[key]
    return class_label

def main():
    data, labels, feature_names = titanic.read_data("discretized-train.csv")
    test, test_labels, _ = titanic.read_data("discretized-test.csv")

    dataset = data_set(data, labels)
    tree = create_tree(dataset, feature_names)
    # print(dataset[0:10])
    # print_tree(tree)

    successes = 0
    totals = 0

    for i in range(len(test)):
        point = test[i]
        # print "* CLASSIFYING: %s" % (",".join(map(lambda x: str(x), point)))
        tree_label = tree_classify(tree, feature_names, point)
        # print "  --> got %s (should be %s)" % (tree_label, test_labels[i])
        if tree_label == test_labels[i]:
            successes += 1
        totals += 1

    print "accuracy: %2.4f %%" % (float(successes) / float(totals) * 100.0)

if __name__ == "__main__":
    main()
