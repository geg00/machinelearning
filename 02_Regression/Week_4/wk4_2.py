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
training,testing = sales.random_split(.9,seed=1)

l2_small_penalty = 1e-5

# ----------------------------------------------
# Observe overfitting
# ----------------------------------------------
print("*** Observe overfitting")

def plot_data(data):    
    plt.plot(data['X1'],data['Y'],'k.')
    plt.xlabel('x')
    plt.ylabel('y')

def polynomial_features(data, deg):
    data_copy=data.copy()
    for i in range(1,deg):
        data_copy['X'+str(i+1)]=data_copy['X'+str(i)]*data_copy['X1']
    return data_copy

def polynomial_regression(data, deg, l2, verbose=False):
    model = graphlab.linear_regression.create(polynomial_features(data, deg),
                                              target = 'Y',
                                              l2_penalty = l2,
                                              validation_set = None)
    print(model)
    return model

def plot_poly_predictions(data, model):
    plot_data(data)
    xmax = data['X1'].max()
    ymax = data['Y'].max()
    
    # Get the degree of the polynomial
    deg = len(model.coefficients['value'])-1
    
    # Create 200 points in the x axis and compute the predicted value for each point
    x_pred = graphlab.SFrame({'X1':[i/200.0 * xmax for i in range(200)]})
    y_pred = model.predict(polynomial_features(x_pred,deg))
    
    # plot predictions
    plt.plot(x_pred['X1'], y_pred, 'g-', label='degree ' + str(deg) + ' fit')
    plt.legend(loc='upper left')
    plt.axis([0,xmax,0,ymax])

import math
import random
import numpy

def print_coefficients(model):    
    # Get the degree of the polynomial
    deg = len(model.coefficients['value'])-1

    # Get learned parameters as a list
    w = list(model.coefficients['value'])

    # Numpy has a nifty function to print out polynomials in a pretty way
    # (We'll use it, but it needs the parameters in the reverse order)
    print 'Learned polynomial for degree ' + str(deg) + ':'
    w.reverse()
    print numpy.poly1d(w)

# ----------------------------------------------
# Selecting an L2 penalty via cross-validation
# ----------------------------------------------
(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

def extract_segment(data, k, i):
    n = len(data)
    start = (n*i)/k
    end = (n*(i+1))/k-1
    return data[start:end+1]

#v = extract_segment(train_valid_shuffled, 10, 0)
#print(v['id'].head())
#v = extract_segment(train_valid_shuffled, 10, 1)
#print(v['id'].head())
#v = extract_segment(train_valid_shuffled, 10, 2)
#print(v['id'].head())
#
#sys.exit(0)

validation4 = extract_segment(train_valid_shuffled, 10, 3)
#validation4.head()

print int(round(validation4['price'].mean(), 0))
print "should be 536,234."

def extract_train(data, k, i):
    n = len(data)
    start = (n*i)/k
    end = (n*(i+1))/k-1
    first_two = data[0:start]
    last_two = data[end+1:n]
    return first_two.append(last_two)

train4 = extract_train(train_valid_shuffled, 10, 3)
#train4.head()

print int(round(train4['price'].mean(), 0))
print "should be 539,450."

def k_fold_cross_validation(k, l2_pena, data, output_name, features_list, verbose=False):
    degree = 15
    rss_sum = 0.
    for i in range(0, k):
        validation = extract_segment(data, k, i)
        training = extract_train(data, k, i)

        poly_data = polynomial_sframe(training['sqft_living'], degree)
        my_features = poly_data.column_names()
#        print(my_features)
        poly_data['price'] = training['price']
        model = graphlab.linear_regression.create(poly_data,
                                                  target = 'price',
                                                  features = my_features,
                                                  l2_penalty = l2_pena,
                                                  validation_set = None,
                                                  verbose = False)
#        model.get("coefficients").print_rows(num_rows=20)

        # validation
        poly_validation = polynomial_sframe(validation[features_list[0]], degree)
        rss = get_residual_sum_of_squares(model, poly_validation, validation[output_name])

        rss_sum += rss
        print("  Segment %d of %d: l2_pena = %f, avg[train X1 = %f, train Y = %f, validation X1 = %f], RSS = %f" % (i, k, l2_pena, training['sqft_living'].mean(), training['price'].mean(), validation['sqft_living'].mean(), rss))
    print("%d-folding, Avg. RSS = %f, L2 penalty = %f" % (k, (rss_sum/k), l2_pena))

#k_fold_cross_validation(10, 0, train_valid_shuffled, 'price', ['sqft_living'])
import numpy as np
#for l2_penalty in np.logspace(1, 7, num=13):
#    k_fold_cross_validation(10, l2_penalty, train_valid_shuffled, 'price', ['sqft_living'])

l2_best_penalty = 1000.
k = 10
degree = 15
poly_data = polynomial_sframe(training['sqft_living'], degree)
my_features = poly_data.column_names()
poly_data['price'] = training['price']
model = graphlab.linear_regression.create(poly_data,
                                          target = 'price',
                                          features = my_features,
                                          l2_penalty = l2_best_penalty,
                                          validation_set = None,
                                          verbose = False)

poly_testing = polynomial_sframe(testing['sqft_living'], degree)
rss = get_residual_sum_of_squares(model, poly_testing, testing['price'])
print("%d-folding, Avg. RSS = %f, L2 penalty = %f" % (k, rss, l2_best_penalty))
