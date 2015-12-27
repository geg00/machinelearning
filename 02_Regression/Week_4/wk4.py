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

# ----------------------------------------------
# Observe overfitting
# ----------------------------------------------
print("*** Observe overfitting")

(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

import sys
sys.path.append("..")

from regression import polynomial_sframe
from regression import get_residual_sum_of_squares

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
                                              target='Y',
                                              l2_penalty=l2,
                                              l1_penalty=0.,
                                              validation_set=None,
                                              verbose=verbose)
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

data = graphlab.SFrame({'X1':set_4['sqft_living'],'Y':set_4['price']})
model_4 = polynomial_regression(data, deg=15, l2=l2_small_penalty)
model_4.get("coefficients").print_rows(num_rows=20)
#plot_poly_predictions(data,model_4)
print_coefficients(model_4)

# ----------------------------------------------
# Ridge regression comes to rescue
# ----------------------------------------------
print("*** Ridge regression comes to rescue")

data = graphlab.SFrame({'X1':set_4['sqft_living'],'Y':set_4['price']})
model_4 = polynomial_regression(data, deg=15, l2=1e5)
model_4.get("coefficients").print_rows(num_rows=20)
#plot_poly_predictions(data,model_4)
print_coefficients(model_4)
