#!/usr/bin/env python2.7

import sys
sys.path.append("..") 

import graphlab

from regression import simple_linear_regression
from regression import get_regression_predictions
from regression import get_residual_sum_of_squares
from regression import inverse_regression_predictions

sales = graphlab.SFrame('kc_house_data.gl/')

train_data,test_data = sales.random_split(.8,seed=0)

print(train_data.head())

print("*** sqft to price")
sqft_intercept,sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)

est_price = get_regression_predictions(2650, sqft_intercept, sqft_slope)

print "Estimated price for 2650 sqft: " + str(est_price)

rss = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'],sqft_intercept,sqft_slope)
print "RSS: " + str(rss)

est_sqft = inverse_regression_predictions(800000, sqft_intercept, sqft_slope)

print "Estimated sqft for $800,000: " + str(est_sqft)

print("*** Bedrooms to price")
bedroom_intercept,bedroom_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])

print "Intercept: " + str(bedroom_intercept)
print "Slope: " + str(bedroom_slope)

rss = get_residual_sum_of_squares(train_data['bedrooms'], train_data['price'],bedroom_intercept,bedroom_slope)
print "RSS: " + str(rss)

print("*** Comparison between sqft and bedrooms with test data")
rss = get_residual_sum_of_squares(test_data['sqft_living'],test_data['price'],sqft_intercept,sqft_slope)
print "RSS with sqft: " + str(rss)

rss = get_residual_sum_of_squares(test_data['bedrooms'],test_data['price'],bedroom_intercept,bedroom_slope)
print "RSS with bedroom: " + str(rss)

