#!/usr/bin/env python2.7

import graphlab

from math import log, sqrt

import sys
sys.path.append("..")
from regression import get_residual_sum_of_squares
import numpy as np

sales = graphlab.SFrame('kc_house_data.gl/')

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to float, before creating a new feature.
sales['floors'] = sales['floors'].astype(float) 
sales['floors_square'] = sales['floors']*sales['floors']

# -------------------------------------
# Learn regression weights with L1 penalty
# -------------------------------------
print("*** Learn regression weights with L1 penalty")
all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = graphlab.linear_regression.create(sales, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty=1e10,
                                              verbose=False)

coef = model_all.get("coefficients")
print coef
print coef[coef["value"]!=0]
print "Number of non-zero coefficients: %d" % len(coef[coef["value"]!=0])

# -------------------------------------
# Selecting an L1 penalty
# -------------------------------------
print("*** Selecting an L1 penalty")
(training_and_validation, testing) = sales.random_split(.9,seed=1) # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1) # split training into train and validate

result = []
lowest_rss = None
best_l1_penalty = None
best_model = None
for l1_penalty in np.logspace(1,7,num=13):
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty=l1_penalty,
                                              verbose=False)
    rss = get_residual_sum_of_squares(model, validation, validation['price'])
    print("l1_penalty = %f, rss = %f" % (l1_penalty, rss))
    result.append("l1_penalty = %f, rss = %f" % (l1_penalty, rss))
    if lowest_rss is None:
        lowest_rss = rss
        best_l1_penalty = l1_penalty
        best_model = model
    if rss < lowest_rss:
        lowest_rss = rss
        best_l1_penalty = l1_penalty
        best_model = model

for r in result:
    print(r)

print "best l1_penalty = %f" % best_l1_penalty

rss = get_residual_sum_of_squares(best_model, testing, testing['price'])

print("testing: l1_penalty = %f, rss = %f" % (best_l1_penalty, rss))
coef = best_model.get("coefficients")
nonzero_coef = coef[coef["value"]!=0]
print "Number of non-zero coefficients: %d" % len(nonzero_coef)
print nonzero_coef

# -------------------------------------
# Limit the number of nonzero weights
# -------------------------------------
print("*** Limit the number of nonzero weights")

max_nonzeros = 7

l1_penalty_min = None
l1_penalty_max = None
l1_penalty_values = np.logspace(8, 10, num=20)
for l1_penalty in l1_penalty_values:
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty=l1_penalty,
                                              verbose=False)
    coef = model.get("coefficients")
#    print coef[coef["value"]!=0]
    nonzero_coef = coef[coef["value"]!=0]
    print("L1 penalty: %f" % l1_penalty)
    print("  Number of non-zero coefficients: %d" % len(nonzero_coef["value"]))

    if len(nonzero_coef["value"]) > max_nonzeros:
        l1_penalty_min = l1_penalty

    if l1_penalty_max is None and len(nonzero_coef["value"]) < max_nonzeros:
        l1_penalty_max = l1_penalty

print("Max L1 penalty: %f" % l1_penalty_max)
print("Min L1 penalty: %f" % l1_penalty_min)

# -------------------------------------
# Exploring the narrow range of values to find the solution with the right number of non-zeros that has lowest RSS on the validation set
# -------------------------------------
print("*** Exploring the narrow range of values to find the solution with the right number of non-zeros that has lowest RSS on the validation set")

l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)

result = []
lowest_rss = None
best_l1_penalty = None
best_model = None
for l1_penalty in l1_penalty_values:
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty=l1_penalty, verbose=False)

    coef = model.get("coefficients")
#    print coef[coef["value"]!=0]
    nonzero_coef = coef[coef["value"]!=0]
    print("L1 penalty: %f" % l1_penalty)
    print("  Number of non-zero coefficients: %d" % len(nonzero_coef["value"]))
    sparsity = len(coef["value"]) - len(nonzero_coef["value"])
    print("  Sparsity: %d" % sparsity)

    if len(nonzero_coef["value"]) == max_nonzeros:
        rss = get_residual_sum_of_squares(model, validation, validation['price'])
        print("  Validation RSS = %f" % rss)
        if lowest_rss is None:
            lowest_rss = rss
            best_l1_penalty = l1_penalty
            best_model = model
        if rss < lowest_rss:
            lowest_rss = rss
            best_l1_penalty = l1_penalty
            best_model = model

for r in result:
    print(r)

print "Best L1 penalty = %f" % best_l1_penalty
print "RSS = %f" % lowest_rss

rss = get_residual_sum_of_squares(best_model, testing, testing['price'])

print("testing: l1_penalty = %f, rss = %f" % (best_l1_penalty, rss))
coef = best_model.get("coefficients")
print coef[coef["value"]!=0]
