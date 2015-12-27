#!/usr/bin/env python2.7

import sys
sys.path.append("..") 

import graphlab
from regression import polynomial_sframe
from regression import get_residual_sum_of_squares

# ----------------------------------------------
# Polynomial regression, revisited
# ----------------------------------------------
print("*** Polynomial regression, revisited")
sales = graphlab.SFrame('kc_house_data.gl/')
sales = sales.sort(['sqft_living','price'])

# split the data set into training, validation and testing.
#training_and_validation,testing = sales.random_split(.9,seed=1)
#training,validation = training_and_validation.random_split(.5,seed=1)
training,testing = sales.random_split(.9,seed=1)

l2_small_penalty = 1e-5

degree = 15
poly_data = polynomial_sframe(training['sqft_living'], degree)
my_features = poly_data.column_names()
print(my_features)
poly_data['price'] = training['price']
model = graphlab.linear_regression.create(poly_data,
                                          target = 'price',
                                          features = my_features,
                                          l2_penalty = l2_small_penalty,
                                          validation_set = None)
model.get("coefficients").print_rows(num_rows=20)

sys.exit(0)

# ----------------------------------------------
# Observe overfitting
# ----------------------------------------------
print("*** Observe overfitting")

(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

def fit(data):
    degree = 15
    poly_data = polynomial_sframe(data['sqft_living'], degree)
    my_features = poly_data.column_names()
    print(my_features)
    poly_data['price'] = data['price']
    model = graphlab.linear_regression.create(poly_data,
                                              target = 'price',
                                              features = my_features,
                                              l2_penalty = l2_small_penalty,
                                              validation_set = None)
    model.get("coefficients").print_rows(num_rows=20)

    poly_validation = polynomial_sframe(validation['sqft_living'], degree)
    rss = get_residual_sum_of_squares(model, poly_validation, validation['price'])

fit(set_1)
fit(set_2)
fit(set_3)
fit(set_4)
