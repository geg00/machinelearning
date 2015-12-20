#!/usr/bin/env python2.7

import graphlab
import sys

sales = graphlab.SFrame('kc_house_data.gl/')
sales = sales.sort(['sqft_living', 'price'])

# -----------------------------------------
# Polynomial_sframe function
# -----------------------------------------
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

def get_residual_sum_of_squares(model, data, outcome):
    RSS = None
    data['prediction'] = model.predict(data)
    data['error'] = outcome - data['prediction']
    error_sq = data['error'] * data['error']
    RSS = error_sq.sum()
    return(RSS)

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
