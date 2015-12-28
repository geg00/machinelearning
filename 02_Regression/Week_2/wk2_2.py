#!/usr/bin/env python2.7

import sys
sys.path.append("..") 

import sys
import graphlab

sales = graphlab.SFrame('kc_house_data.gl/')

import numpy as np # note this allows us to refer to numpy as np instead 

import unittest
from regression import get_numpy_data
from regression import predict_output
from regression import feature_derivative
from regression import regression_gradient_descent

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
# the [] around 'sqft_living' makes it a list

print "example_features"
print len(example_features)
print example_features[0:3,:]
# this accesses the first row of the data the ':' indicates 'all columns'
print "example_output"
print len(example_output)
print example_output[0]
# and the corresponding output

# Predicting output given regression weights
print("*** Predicting output given regression weights")

my_weights = np.array([1., 1.]) # the example weights
print(my_weights)

my_features = example_features[0,] # we'll use the first data point
print(my_features)

predicted_value = np.dot(my_features, my_weights)
print "predicted_value"
print predicted_value

test_predictions = predict_output(example_features, my_weights)
print "test_predictions[0] should be 1181.0"
print test_predictions[0] # should be 1181.0
print "test_predictions[1] should be 2571.0"
print test_predictions[1] # should be 2571.0

# Computing the Derivative
print("*** Computing the Derivative")

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
my_weights = np.array([0., 0.]) # this makes all the predictions 0
test_predictions = predict_output(example_features, my_weights) 

# just like SFrames 2 numpy arrays can be elementwise subtracted with '-': 
errors = test_predictions - example_output # prediction errors in this case is just the -example_output
feature = example_features[:,0] # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"

derivative = feature_derivative(errors, feature)

print "derivative"
print derivative
print "-np.sum(example_output)*2"
print -np.sum(example_output)*2 # should be the same as derivative


# Gradient Descent
print "*** Gradient Descent"
# recall that the magnitude/length of a vector [g[0], g[1], g[2]] is sqrt(g[0]^2 + g[1]^2 + g[2]^2)

# -----------------------------------
# Running the Gradient Descent as Simple Regression
# -----------------------------------
print "*** Running the Gradient Descent as Simple Regression"

train_data,test_data = sales.random_split(.8,seed=0)

# let's test out the gradient descent
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)

print "simple_weights"
print(simple_weights)

(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

simple_predictions = predict_output(test_simple_feature_matrix, simple_weights)

print("1st house estimated price = %f" % simple_predictions[0])
print("1st house actual price = %f" % output[0])
print("1st house error = %f" % (simple_predictions[0] - output[0]))

r = test_output - simple_predictions
RSS = (r**2).sum()
print("RSS = %f" % RSS)

# -----------------------------------
# Running a multiple regression
# -----------------------------------
print "*** Running a multiple regression"

train_data,test_data = sales.random_split(.8,seed=0)

model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

model_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print "model_weights"
print(model_weights)

(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

#model_weights = regression_gradient_descent(test_feature_matrix, test_output, initial_weights, step_size, tolerance)
#print "model_weights"
#print(model_weights)

#sys.exit(0)

model_predictions = predict_output(test_feature_matrix, model_weights)

print("1st house estimated price = %f" % model_predictions[0])
print("1st house actual price = %f" % output[0])
print("1st house error = %f" % (model_predictions[0] - output[0]))

r = test_output - model_predictions
RSS = (r**2).sum()
print("RSS = %f" % RSS)

