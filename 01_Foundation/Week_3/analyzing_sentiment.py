#!/usr/local/bin/python

import graphlab

products = graphlab.SFrame('amazon_baby.gl/')

products['word_count'] = graphlab.text_analytics.count_words(products['review'])

print(products.head())


