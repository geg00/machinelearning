#!/usr/bin/env python2.7

import graphlab
import numpy as np # note this allows us to refer to numpy as np instead 
import unittest

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

def get_residual_sum_of_squares(input_feature, output, intercept, slope, verbose=False):
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

def get_residual_sum_of_squares_(model, data, outcome):
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

    def test_get_residual_sum_of_squares_001(self):
        sales = graphlab.SFrame('kc_house_data.gl/')
        train_data,test_data = sales.random_split(.8,seed=0)

        sqft_intercept = -47116.07657494047
        sqft_slope = 281.9588385676974

        rss = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'],sqft_intercept,sqft_slope)

        self.assertEqual(1201918356321967.5, rss)

    def test_inverse_regression_predictions_001(self):
        sqft_intercept = -47116.07657494047
        sqft_slope = 281.9588385676974

        est_sqft = inverse_regression_predictions(800000,sqft_intercept,sqft_slope)

        self.assertEqual(3004.3962476159463, est_sqft)

if __name__ == '__main__':
    unittest.main()
