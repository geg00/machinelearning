#!/usr/bin/env python2.7

import sys
sys.path.append("..")

import graphlab
from regression import polynomial_sframe
from regression import get_residual_sum_of_squares

sales = graphlab.SFrame('kc_house_data.gl/')
sales = sales.sort(['sqft_living', 'price'])

# -----------------------------------------
# Selecting a Polynomial Degree
# -----------------------------------------fe
print("*** Selecting a Polynomial Degree")

training_and_validation,testing = sales.random_split(.9,seed=1)
training,validation = training_and_validation.random_split(.5,seed=1)
print(len(sales))
print(len(training))
print(len(validation))
print(len(testing))

foo = ""
lowest_rss = None
lowest_degree = None
lowest_model = None
for degree in range(1, 15+1):
    print("----------------------------------------------")
    print("Estimating for Degree %d" % degree)
    print("----------------------------------------------")
    poly_data = polynomial_sframe(training['sqft_living'], degree)
    my_features = poly_data.column_names()
    print(my_features)
    poly_data['price'] = training['price']
    model = graphlab.linear_regression.create(poly_data, target = 'price', features = my_features, validation_set = None)
    model.get("coefficients").print_rows(num_rows=20)

    poly_validation = polynomial_sframe(validation['sqft_living'], degree)
    rss = get_residual_sum_of_squares(model, poly_validation, validation['price'])
    if lowest_rss is None or lowest_rss > rss:
        lowest_rss = rss
        lowest_degree = degree
        lowest_model = model
    foo = foo + "Degree %d, RSS %f\n" % (degree, rss)

print(foo)
print("Lowest RSS %f at degree %d" % (lowest_rss, lowest_degree))

poly_testing = polynomial_sframe(testing['sqft_living'], lowest_degree)
rss = get_residual_sum_of_squares(lowest_model, poly_testing, testing['price'])

print("RSS %f on Testing data" % (rss))
