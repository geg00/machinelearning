#!/usr/bin/env python2.7

import sys
sys.path.append("..") 

import graphlab
import numpy as np # note this allows us to refer to numpy as np instead 

from regression import get_numpy_data
from regression import predict_output

import unittest

sales = graphlab.SFrame('kc_house_data.gl/')

# -------------------------------
# Computing the Derivative
# -------------------------------
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant, verbose=False):
    derivative = None

    if verbose is True:
        print("errors: %s [%d]" % (str(errors), len(errors)))
        print("feature: %s [%d]" % (str(feature), len(feature)))
        print("weight: %s" % str(weight))
        print("l2_penalty = %f" % l2_penalty)

    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if feature_is_constant is True:
        dot_errors_and_feature = errors * feature
        if verbose is True:
            print("dot_errors_and_feature: %s [%d]" % (str(dot_errors_and_feature), len(dot_errors_and_feature)))
        derivative = 2*dot_errors_and_feature.sum()

    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
        dot_errors_and_feature = errors * feature
        if verbose is True:
            print("dot_errors_and_feature: %s [%d]" % (str(dot_errors_and_feature), len(dot_errors_and_feature)))
        derivative = 2*dot_errors_and_feature.sum() + 2*l2_penalty*weight

    return derivative

# (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
# my_weights = np.array([1., 10.])
# test_predictions = predict_output(example_features, my_weights) 
# errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
# print feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
# print np.sum(errors*example_features[:,1])*2+20.
# print ''

# next two lines should print the same values
# print feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
# print np.sum(errors)*2.

# -------------------------------
# Gradient Descent
# -------------------------------
def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100, verbose=False):
    weights = np.array(initial_weights) # make sure it's a numpy array

    if verbose is True:
        print("initial weights: %s" % str(weights))

    #while not reached maximum number of iterations:
    for iter in range(0, max_iterations):
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        test_predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = test_predictions - output

        for i in xrange(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            #(Remember: when i=0, you are computing the derivative of the constant!)
            if i == 0:
                is_constant = True
            else:
                is_constant = False
            derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, is_constant)

            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - derivative * step_size

        if verbose is True:
            print("iteration = %d, derivative = %f, weights = %s" % (iter, derivative, str(weights)))

    return weights

# -------------------------------
# Visualizing effect of L2 penalty
# -------------------------------
simple_features = ['sqft_living']
my_output = 'price'
train_data,test_data = sales.random_split(.8,seed=0)

(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations = 1000

l2_penalty = 0.
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix,
                                                             output,
                                                             initial_weights,
                                                             step_size,
                                                             l2_penalty,
                                                             max_iterations)
print("simple_weights_0_penalty = %s" % str(simple_weights_0_penalty))
# simple_weights_0_penalty = [ 262.91714394  262.91714394]

l2_penalty = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix,
                                                                output,
                                                                initial_weights,
                                                                step_size,
                                                                l2_penalty,
                                                                max_iterations)
print("simple_weights_high_penalty = %s" % str(simple_weights_high_penalty))
# simple_weights_high_penalty = [ 124.57082656  124.57082656]

# Compute the RSS on the TEST data for the following three sets of weights:
# 
#     The initial weights (all zeros)
#     The weights learned with no regularization
#     The weights learned with high regularization
# 
# Which weights perform best?

test_data['price_1'] = predict_output(simple_test_feature_matrix, initial_weights)
test_data['price_2'] = predict_output(simple_test_feature_matrix, simple_weights_0_penalty)
test_data['price_3'] = predict_output(simple_test_feature_matrix, simple_weights_high_penalty)

print(test_data['id','sqft_living','price','price_1','price_2','price_3'].head())

def get_residual_sum_of_squares2(outcome, prediction):
    RSS = None
    error = outcome - prediction
    error_sq = error * error
    RSS = error_sq.sum()
    return(RSS)

print '[simple regression]'
print "RSS of initial weights:             %f" % get_residual_sum_of_squares2(test_data['price'], test_data['price_1'])
print "RSS of simple_weights_0_penalty:    %f" % get_residual_sum_of_squares2(test_data['price'], test_data['price_2'])
print "RSS of simple_weights_high_penalty: %f" % get_residual_sum_of_squares2(test_data['price'], test_data['price_3'])

# -------------------------------
# Running a multiple regression with L2 penalty
# -------------------------------
model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000

l2_penalty = 0.
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix,
                                                               output,
                                                               initial_weights,
                                                               step_size,
                                                               l2_penalty,
                                                               max_iterations,
                                                               verbose=False)
print("multiple_weights_0_penalty = %s" % str(multiple_weights_0_penalty))

l2_penalty = 1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix,
                                                                  output,
                                                                  initial_weights,
                                                                  step_size,
                                                                  l2_penalty,
                                                                  max_iterations,
                                                                  verbose=False)
print("multiple_weights_high_penalty = %s" % str(multiple_weights_high_penalty))

test_data['price_1'] = predict_output(test_feature_matrix, initial_weights)
test_data['price_2'] = predict_output(test_feature_matrix, multiple_weights_0_penalty)
test_data['price_3'] = predict_output(test_feature_matrix, multiple_weights_high_penalty)

print(test_data['id','sqft_living','sqft_living15','price','price_1','price_2','price_3'].head())

print '[multiple regression]'
print "RSS of initial weights:               %f" % get_residual_sum_of_squares2(test_data['price'], test_data['price_1'])
print "RSS of multiple_weights_0_penalty:    %f" % get_residual_sum_of_squares2(test_data['price'], test_data['price_2'])
print "RSS of multiple_weights_high_penalty: %f" % get_residual_sum_of_squares2(test_data['price'], test_data['price_3'])

class TestWk4_3(unittest.TestCase):
    def test_feature_derivative_ridge_001(self):
        (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
        my_weights = np.array([1., 10.])
        test_predictions = predict_output(example_features, my_weights) 
        errors = test_predictions - example_output # prediction errors

        a = feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
        b = np.sum(errors*example_features[:,1])*2+20.
        self.assertEqual(b, a)

    def test_feature_derivative_ridge_002(self):
        (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
        my_weights = np.array([1., 10.])
        test_predictions = predict_output(example_features, my_weights) 
        errors = test_predictions - example_output # prediction errors

        a = feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
        b = np.sum(errors)*2.
        self.assertEqual(b, a)

    def test_ridge_regression_gradient_descent_001(self):
        simple_features = ['sqft_living']
        my_output = 'price'
        train_data,test_data = sales.random_split(.8,seed=0)

        (simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
        (simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

        initial_weights = np.array([0., 0.])
        step_size = 1e-12
        max_iterations = 1000

        l2_penalty = 0.
        simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix,
                                                                     output,
                                                                     initial_weights,
                                                                     step_size,
                                                                     l2_penalty,
                                                                     max_iterations)
        a = np.array([-1.63113501e-01,2.63024369e+02])
        self.assertEqual(str(a), str(simple_weights_0_penalty))

    def test_ridge_regression_gradient_descent_002(self):
        simple_features = ['sqft_living']
        my_output = 'price'
        train_data,test_data = sales.random_split(.8,seed=0)

        (simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
        (simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

        initial_weights = np.array([0., 0.])
        step_size = 1e-12
        max_iterations = 1000

        l2_penalty = 1e11
        simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix,
                                                                        output,
                                                                        initial_weights,
                                                                        step_size,
                                                                        l2_penalty,
                                                                        max_iterations)
        a = np.array([9.76730383,124.57217565])
        self.assertEqual(str(a), str(simple_weights_high_penalty))

    def test_ridge_regression_gradient_descent_003(self):
        model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
        my_output = 'price'
        train_data,test_data = sales.random_split(.8,seed=0)

        (feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
        (test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
        
        initial_weights = np.array([0.0,0.0,0.0])
        step_size = 1e-12
        max_iterations = 1000
        
        l2_penalty = 0.
        multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix,
                                                                       output,
                                                                       initial_weights,
                                                                       step_size,
                                                                       l2_penalty,
                                                                       max_iterations)
        a = np.array([-0.35743482,243.0541689,22.41481594])
        self.assertEqual(str(a), str(multiple_weights_0_penalty))

    def test_ridge_regression_gradient_descent_004(self):
        model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
        my_output = 'price'
        train_data,test_data = sales.random_split(.8,seed=0)

        (feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
        (test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
        
        initial_weights = np.array([0.0,0.0,0.0])
        step_size = 1e-12
        max_iterations = 1000
        
        l2_penalty = 1e11
        multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix,
                                                                       output,
                                                                       initial_weights,
                                                                       step_size,
                                                                       l2_penalty,
                                                                       max_iterations)
        a = np.array([6.7429658,91.48927361,78.43658768])
        self.assertEqual(str(a), str(multiple_weights_high_penalty))

if __name__ == '__main__':
    unittest.main()
