#!/usr/bin/env python2.7

import graphlab

sales = graphlab.SFrame('kc_house_data.gl/')

train_data,test_data = sales.random_split(.8,seed=0)

print(train_data.head())

def simple_linear_regression(input_feature, output):
    intercept = 0.0
    slope = 0.0

    print("input feature: " + str(input_feature.head()))
    print("output: " + str(output.head()))

    x = input_feature
    y = output
    yx = y * x
    xx = x * x
    N = input_feature.size()

    w1 = (yx.sum() - y.sum() * x.sum() / N) / ( xx.sum() - x.sum() * x.sum() / N)
    print("w1 = %f" % w1)

    w0 = y.mean() - w1 * x.mean()
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

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    y = output
    x = input_feature
    w0 = intercept
    w1 = slope

    pred_y = w0 + x * w1
    print("Predicted Y: " + str(pred_y.head()))
    r = y - pred_y
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
