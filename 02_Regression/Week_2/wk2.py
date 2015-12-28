#!/usr/bin/env python2.7

import sys
sys.path.append("..") 

import graphlab
from regression import get_residual_sum_of_squares

sales = graphlab.SFrame('kc_house_data.gl/')

# Split data into training and testing.
train_data,test_data = sales.random_split(.8,seed=0)

# Learning a multiple regression model
example_features = ['sqft_living', 'bedrooms', 'bathrooms']
example_model = graphlab.linear_regression.create(train_data, target = 'price',
                                                  features = example_features, 
                                                  validation_set = None)

example_weight_summary = example_model.get("coefficients")
print example_weight_summary

# Making Predictions
example_predictions = example_model.predict(train_data)
# should be 271789.505878
print example_predictions[0]

rss_example_train = get_residual_sum_of_squares(example_model,
                                                test_data,
                                                test_data['price'])
print rss_example_train # should be 2.7376153833e+14

# Create some new features
from math import log

train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)

train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']

train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))

train_data['lat_plus_long'] = train_data['lat'] + train_data['long']
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

for f in ['bedrooms_squared', 'bed_bath_rooms', 'log_sqft_living', 'lat_plus_long']:
    print("%s: avg %f" % (f, test_data[f].mean()))

# Learning Multiple Models
model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

model_1 = graphlab.linear_regression.create(train_data, target = 'price',
                                            features = model_1_features,
                                            validation_set = None)

model_2 = graphlab.linear_regression.create(train_data, target = 'price',
                                            features = model_2_features,
                                            validation_set = None)

model_3 = graphlab.linear_regression.create(train_data, target = 'price',
                                            features = model_3_features,
                                            validation_set = None)

print "*** Model 1"
model_1_weight_summary = model_1.get("coefficients")
print model_1_weight_summary

print "*** Model 2"
model_2_weight_summary = model_2.get("coefficients")
print model_2_weight_summary

print "*** Model 3"
model_3_weight_summary = model_3.get("coefficients")
print model_3_weight_summary

# Comparing multiple models
# RSS of train data
rss_model_1 = get_residual_sum_of_squares(model_1,
                                          train_data,
                                          train_data['price'])
print("RSS of model 1 on training data: %f" % rss_model_1)

rss_model_2 = get_residual_sum_of_squares(model_2,
                                          train_data,
                                          train_data['price'])
print("RSS of model 2 on training data: %f" % rss_model_2)

rss_model_3 = get_residual_sum_of_squares(model_3,
                                          train_data,
                                          train_data['price'])
print("RSS of model 3 on training data: %f" % rss_model_3)

# RSS of test data
rss_model_1 = get_residual_sum_of_squares(model_1,
                                          test_data,
                                          test_data['price'])
print("RSS of model 1 on test data: %f" % rss_model_1)

rss_model_2 = get_residual_sum_of_squares(model_2,
                                          test_data,
                                          test_data['price'])
print("RSS of model 2 on test data: %f" % rss_model_2)

rss_model_3 = get_residual_sum_of_squares(model_3,
                                          test_data,
                                          test_data['price'])
print("RSS of model 3 on test data: %f" % rss_model_3)

