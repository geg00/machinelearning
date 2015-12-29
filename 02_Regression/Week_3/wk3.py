#!/usr/bin/env python2.7

import sys
sys.path.append("..")

import graphlab
from regression import polynomial_sframe

tmp = graphlab.SArray([1., 2., 3.])
tmp_cubed = tmp.apply(lambda x: x**3)
print tmp
print tmp_cubed

ex_sframe = graphlab.SFrame()
ex_sframe['power_1'] = tmp
print ex_sframe

# -----------------------------------------
# Polynomial_sframe function
# -----------------------------------------
print("*** Polynomial_sframe function")

print polynomial_sframe(tmp, 3)

# -----------------------------------------
# Visualizing polynomial regression
# -----------------------------------------
print("*** Visualizing polynomial regression")

sales = graphlab.SFrame('kc_house_data.gl/')

sales = sales.sort(['sqft_living', 'price'])

poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target

model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)

#let's take a look at the weights before we plot
model1.get("coefficients")

# import matplotlib.pyplot as plt
# %matplotlib inline

# plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
#         poly1_data['power_1'], model1.predict(poly1_data),'-')

poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)

model2.get("coefficients")

# plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
#         poly2_data['power_1'], model2.predict(poly2_data),'-')
