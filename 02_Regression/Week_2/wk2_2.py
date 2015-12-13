#!/usr/bin/env python2.7

import sys
import graphlab

sales = graphlab.SFrame('kc_house_data.gl/')

import numpy as np # note this allows us to refer to numpy as np instead 

def get_numpy_data(data_sframe, features, output):
    print(data_sframe.head())
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame

    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists

    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]

    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()

    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe[output]

    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)

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

def predict_output(feature_matrix, weights):
    predictions = None

    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
#    print("feature_matrix")
#    print(len(feature_matrix))
#    print("weights")
#    print(len(weights))

    predictions = np.dot(feature_matrix, weights)
#    print("predictions")
#    print(len(predictions))

    return(predictions)

test_predictions = predict_output(example_features, my_weights)
print "test_predictions[0] should be 1181.0"
print test_predictions[0] # should be 1181.0
print "test_predictions[1] should be 2571.0"
print test_predictions[1] # should be 2571.0

# Computing the Derivative
print("*** Computing the Derivative")

def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = np.dot(errors, feature) * 2

    return(derivative)

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
from math import sqrt
# recall that the magnitude/length of a vector [g[0], g[1], g[2]] is sqrt(g[0]^2 + g[1]^2 + g[2]^2)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array
    count = 0
    while not converged:
        print("count = %d" % count)
        print("weights = %s" % str(weights))

        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)

        # compute the errors as predictions - output
        errors = predictions - output

        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[:, i])
            print("  derivative = %f" % derivative)

            # add the squared value of the derivative to the gradient magnitude (for assessing convergence)
            gradient_sum_squares = gradient_sum_squares + (derivative * derivative)

            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - (derivative * step_size)
            
        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        print("tolerance = %f, gradient_magnitude = %f" % (tolerance, gradient_magnitude))
        if gradient_magnitude < tolerance:
            converged = True
        count += 1
        if gradient_magnitude == float("inf"):
            sys.exit(0)

    return(weights)

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
