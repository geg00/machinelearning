#!/usr/bin/env python2.7

import graphlab
import numpy as np # note this allows us to refer to numpy as np instead 
import unittest
from math import sqrt

# ---------------------------------------------
# Week 1: Simple Linear Regression
# ---------------------------------------------
def simple_linear_regression(input_feature, output, verbose=False):
    intercept = 0.0
    slope = 0.0

    if verbose is True:
        print("input feature: " + str(input_feature.head()))
        print("output: " + str(output.head()))

    x = input_feature
    y = output
    yx = y * x
    xx = x * x
    N = input_feature.size()

    w1 = (yx.sum() - y.sum() * x.sum() / N) / ( xx.sum() - x.sum() * x.sum() / N)
    if verbose is True:
        print("w1 = %f" % w1)

    w0 = y.mean() - w1 * x.mean()
    if verbose is True:
        print("w0 = %f" % w0)

    # compute the mean of  input_feature and output
    # compute the product of the output and the input_feature and its mean
    # compute the squared value of the input_feature and its mean
    # use the formula for the slope
    # use the formula for the intercept
    
    intercept = w0
    slope = w1

    return(intercept, slope)

def get_regression_predictions(input_feature, intercept, slope):
    x = input_feature
    w0 = intercept
    w1 = slope

    y = w0 + w1 * x

    predicted_output = y
    return(predicted_output)

def get_residual_sum_of_squares_1(input_feature, output, intercept, slope, verbose=False):
    y = output
    x = input_feature
    w0 = intercept
    w1 = slope

    pred_y = w0 + x * w1
    if verbose is True:
        print("Predicted Y: " + str(pred_y.head()))
    r = y - pred_y
    if verbose is True:
        print("Redidual: " + str(r.head()))
    
    RSS = (r * r).sum()
    return(RSS)

def inverse_regression_predictions(output, intercept, slope):
    w0 = intercept
    w1 = slope
    y = output

    x = (y - w0) / w1

    estimated_input = x
    return(estimated_input)


# ---------------------------------------------
# Week 2: Multiple Regression
# ---------------------------------------------
def get_residual_sum_of_squares(model, data, outcome):
    RSS = None
    data['prediction'] = model.predict(data)
    data['error'] = outcome - data['prediction']
    error_sq = data['error'] * data['error']
    RSS = error_sq.sum()
    return(RSS)

def get_numpy_data(data_sframe, features, output, verbose=False):
    if verbose is True:
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

def predict_output(feature_matrix, weights):
    predictions = None
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)


def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = np.dot(errors, feature) * 2

    return(derivative)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance, verbose=False):
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array
    count = 0
    while not converged:
        if verbose is True:
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
            if verbose is True:
                print("  derivative = %f" % derivative)

            # add the squared value of the derivative to the gradient magnitude (for assessing convergence)
            gradient_sum_squares = gradient_sum_squares + (derivative * derivative)

            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - (derivative * step_size)
            
        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if verbose is True:
            print("tolerance = %f, gradient_magnitude = %f" % (tolerance, gradient_magnitude))
        if gradient_magnitude < tolerance:
            converged = True
        count += 1
        if gradient_magnitude == float("inf"):
            sys.exit(0)

    return(weights)


# ---------------------------------------------
# Week 3: Assessing Performance
# ---------------------------------------------
def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    if degree < 0:
        print("ERROR: degree should be >=0.")
        sys.exit(1)

    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature

    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature.apply(lambda x: x**power)

    return poly_sframe


class TestWeek1(unittest.TestCase):
    def test_simple_linear_regression_001(self):
        sales = graphlab.SFrame('kc_house_data.gl/')
        train_data,test_data = sales.random_split(.8,seed=0)

        sqft_intercept,sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

        self.assertEqual(-47116.07657494047, sqft_intercept)
        self.assertEqual(281.9588385676974, sqft_slope)

    def test_get_regression_predictions_001(self):
        sqft_intercept = -47116.07657494047
        sqft_slope = 281.9588385676974

        est_price = get_regression_predictions(2650, sqft_intercept, sqft_slope)

        self.assertEqual(700074.8456294576, est_price)

    def test_get_residual_sum_of_squares_1_001(self):
        sales = graphlab.SFrame('kc_house_data.gl/')
        train_data,test_data = sales.random_split(.8,seed=0)

        sqft_intercept = -47116.07657494047
        sqft_slope = 281.9588385676974

        rss = get_residual_sum_of_squares_1(train_data['sqft_living'], train_data['price'],sqft_intercept,sqft_slope)

        self.assertEqual(1201918356321967.5, rss)

    def test_inverse_regression_predictions_001(self):
        sqft_intercept = -47116.07657494047
        sqft_slope = 281.9588385676974

        est_sqft = inverse_regression_predictions(800000,sqft_intercept,sqft_slope)

        self.assertEqual(3004.3962476159463, est_sqft)

class TestWeek2(unittest.TestCase):
    def test_simple_linear_regression_001(self):
        sales = graphlab.SFrame('kc_house_data.gl/')

        # Split data into training and testing.
        train_data,test_data = sales.random_split(.8,seed=0)

        # Learning a multiple regression model
        example_features = ['sqft_living', 'bedrooms', 'bathrooms']
        example_model = graphlab.linear_regression.create(train_data, target = 'price',
                                                          features = example_features, 
                                                          validation_set = None)

        rss_example_train = get_residual_sum_of_squares(example_model,
                                                        test_data,
                                                        test_data['price'])

        self.assertEqual(273761538330191.1, rss_example_train)

    def test_get_numpy_data_001(self):
        sales = graphlab.SFrame('kc_house_data.gl/')
        (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')

        self.assertEqual(21613, len(example_features))
        a = np.array([[1.00000000e+00,1.18000000e+03],
                      [1.00000000e+00,2.57000000e+03],
                      [1.00000000e+00,7.70000000e+02]])
        self.assertEqual(str(a), str(example_features[0:3,:]))

        self.assertEqual(21613, len(example_output))
        self.assertEqual(221900.0, example_output[0])
        self.assertEqual(325000.0, example_output[len(example_output)-1])

    def test_predict_output_001(self):
        sales = graphlab.SFrame('kc_house_data.gl/')
        (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')

        my_weights = np.array([1., 1.]) # the example weights

        test_predictions = predict_output(example_features, my_weights)

        self.assertEqual(1181.0, test_predictions[0])
        self.assertEqual(2571.0, test_predictions[1])

    def test_feature_derivative_001(self):
        sales = graphlab.SFrame('kc_house_data.gl/')
        (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')

        my_weights = np.array([0., 0.]) # this makes all the predictions 0
        test_predictions = predict_output(example_features, my_weights) 

        errors = test_predictions - example_output # prediction errors in this case is just the -example_output
        feature = example_features[:,0] # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
        derivative = feature_derivative(errors, feature)

        self.assertEqual(-np.sum(example_output)*2, derivative)
        self.assertEqual(-23345850022.0, derivative)

    def test_regression_gradient_descent_001(self):
        sales = graphlab.SFrame('kc_house_data.gl/')
        train_data,test_data = sales.random_split(.8,seed=0)

        # let's test out the gradient descent
        simple_features = ['sqft_living']
        my_output = 'price'
        (simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
        initial_weights = np.array([-47000., 1.])
        step_size = 7e-12
        tolerance = 2.5e7

        simple_weights = regression_gradient_descent(simple_feature_matrix, output,
                                                     initial_weights, step_size, tolerance)

        a = np.array([-46999.88716555,281.91211912])
        self.assertEqual(str(a), str(simple_weights))

    def test_regression_gradient_descent_002(self):
        sales = graphlab.SFrame('kc_house_data.gl/')
        train_data,test_data = sales.random_split(.8,seed=0)

        model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
        my_output = 'price'
        (feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
        initial_weights = np.array([-100000., 1., 1.])
        step_size = 4e-12
        tolerance = 1e9

        model_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)

        a = np.array([-9.99999688e+04,2.45072603e+02,6.52795277e+01])
        self.assertEqual(str(a), str(model_weights))

class TestWeek3(unittest.TestCase):
    def test_spolynomial_sframe_001(self):
        tmp = graphlab.SArray([1., 2., 3.])
        f = polynomial_sframe(tmp, 3)

        self.assertEqual(['power_1', 'power_2', 'power_3'], f.column_names())
        self.assertEqual('[1.0, 2.0, 3.0]', str(f['power_1']))
        self.assertEqual('[1.0, 4.0, 9.0]', str(f['power_2']))
        self.assertEqual('[1.0, 8.0, 27.0]', str(f['power_3']))

if __name__ == '__main__':
    unittest.main()
