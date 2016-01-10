#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import graphlab
import numpy as np # note this allows us to refer to numpy as np instead 
import sys
sys.path.append("..")
from regression import get_numpy_data
from regression import predict_output
from regression import normalize_features
from regression import lasso_coordinate_descent_step
from regression import lasso_cyclical_coordinate_descent
import unittest
import copy

# ---------------------------------------
# Load in house sales data
# ---------------------------------------
print("*** Load in house sales data")

sales = graphlab.SFrame('kc_house_data.gl/')
# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int) 

# ---------------------------------------
# Normalize features
# ---------------------------------------
print("*** Normalize features")

# ---------------------------------------
# Implementing Coordinate Descent with normalized features
# ---------------------------------------
print("*** Implementing Coordinate Descent with normalized features")

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
simple_feature_matrix, norms = normalize_features(simple_feature_matrix)

weights = np.array([1., 4., 1.])
weights

prediction = predict_output(simple_feature_matrix, weights)
prediction

w = weights
# need to normalize output here?
ro = {}
for i in range(0,len(w)):
    feature_i = simple_feature_matrix[:,i]
    tmp = feature_i * (output - prediction + w[i]*feature_i)
    print tmp
    ro[i] = tmp.sum()
print("ro[i]: %s" % str(ro))

# ---------------------------------------
# Single Coordinate Descent Step
# ---------------------------------------
print("*** Single Coordinate Descent Step")

# ---------------------------------------
# Cyclical coordinate descent
# ---------------------------------------
print("*** Cyclical coordinate descent")

def get_rss(predict, output):
    tmp = (predict - output) ** 2
    return tmp.sum()

# ---------------------------------------
# Evaluating LASSO fit with more features
# ---------------------------------------
print("*** Evaluating LASSO fit with more features")


if __name__ == '__main__':
    unittest.main()
