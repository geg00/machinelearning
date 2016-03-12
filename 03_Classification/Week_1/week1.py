#!/usr/bin/env python2.7

from __future__ import division
import graphlab
import math
import string

# -----------------------------------
# Data preperation
print "*** Data preperation"
# -----------------------------------
products = graphlab.SFrame('amazon_baby.gl/')
products

# -----------------------------------
# Build the word count vector for each review
print "*** Build the word count vector for each review"
# -----------------------------------
products[269]

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

review_without_puctuation = products['review'].apply(remove_punctuation)
review_without_puctuation = products['review'].apply(remove_punctuation)

products['word_count'] = graphlab.text_analytics.count_words(review_without_puctuation)

products[269]['word_count']

# -----------------------------------
# Extract sentiments
print "*** Extract sentiments"
# -----------------------------------
len(products)
products = products[products['rating'] != 3]
len(products)

products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products


# -----------------------------------
# Split data into training and test sets
print "*** Split data into training and test sets"
# -----------------------------------
train_data, test_data = products.random_split(.8, seed=1)
print len(train_data)
print len(test_data)

# -----------------------------------
# Train a sentiment classifier with logistic regression
print "*** Train a sentiment classifier with logistic regression"
# -----------------------------------
sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target = 'sentiment',
                                                      features=['word_count'],
                                                      validation_set=None)
sentiment_model

weights = sentiment_model.coefficients
weights.column_names()

positive_weights = weights[weights['value'] >= 0]
negative_weights = weights[weights['value'] < 0]

num_positive_weights = len(positive_weights)
num_negative_weights = len(negative_weights)

print "Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights

# Quiz question: How many weights are >= 0?
# Number of positive weights: 68419

# -----------------------------------
# Making predictions with logistic regression
print "*** Making predictions with logistic regression"
# -----------------------------------
sample_test_data = test_data[10:13]
print sample_test_data['rating']
sample_test_data

sample_test_data[0]['review']

sample_test_data[1]['review']

sample_test_data['score'] = sentiment_model.predict(sample_test_data, output_type='margin')
sample_test_data['score']

# -----------------------------------
# Predicting sentiment
print "*** Predicting sentiment"
# -----------------------------------
sample_test_data['pred_sentiment'] = sample_test_data['score'].apply(lambda score: +1 if score > 0 else -1)
sample_test_data

print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data)

# -----------------------------------
# Probability predictions
print "*** Probability predictions"
# -----------------------------------

from math import exp
sample_test_data['probability'] = sample_test_data['score'].apply(lambda score: 1.0/(1.0+exp(-1.0*score)))
sample_test_data

print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data, output_type='probability')

# Quiz Question: Of the three data points in sample_test_data, which one (first, second, or third) has the lowest probability of being classified as a positive review?
# +----------------+-------------------+
# |     score      |        prob       |
# +----------------+-------------------+
# | 6.73461972706  |   0.998812384838  |
# | -5.73413099676 |  0.0032232681818  |
# | -14.6684604045 | 4.26155799665e-07 |
# +----------------+-------------------+

# -----------------------------------
# Find the most positive (and negative) review
print "*** Find the most positive (and negative) review"
# -----------------------------------
test_data['score'] = sentiment_model.predict(test_data, output_type='margin')
test_data['pred_sentiment'] = test_data['score'].apply(lambda score: +1 if score > 0 else -1)
test_data['probability'] = sentiment_model.predict(test_data, output_type='probability')

positive20 = test_data.topk('probability', k=20)
positive20.print_rows(num_rows=20)

# Quiz Question: Which of the following products are represented in the 20 most positive reviews? [multiple choice]
# +-------------------------------+-------------------------------+--------+
# |              name             |             review            | rating |
# +-------------------------------+-------------------------------+--------+
# | Britax Decathlon Convertib... | I researched a few differe... |  4.0   |
# | Ameda Purely Yours Breast ... | As with many new moms, I c... |  4.0   |
# | Traveling Toddler Car Seat... | I am sure this product wor... |  2.0   |
# | Shermag Glider Rocker Comb... | After searching in stores ... |  4.0   |
# | Cloud b Sound Machine Soot... | First off, I love plush sh... |  5.0   |
# | JP Lizzy Chocolate Ice Cla... | I got this bag as a presen... |  4.0   |
# | Fisher-Price Rainforest Me... | My daughter wasn't able to... |  5.0   |
# | Lilly Gold Sit 'n' Stroll ... | I just completed a two-mon... |  5.0   |
# |  Fisher-Price Deluxe Jumperoo | I had already decided that... |  5.0   |
# | North States Supergate Pre... | I got a couple of these to... |  4.0   |
# |   Munchkin Mozart Magic Cube  | My wife and I have been li... |  4.0   |
# | Britax Marathon Convertibl... | My son began using the Mar... |  5.0   |
# | Wizard Convertible Car Sea... | My son was born big and re... |  5.0   |
# |   Capri Stroller - Red Tech   | First let me say that I wa... |  4.0   |
# | Peg Perego Primo Viaggio C... | We have been using this se... |  5.0   |
# | HALO SleepSack Micro-Fleec... | I love the Sleepsack weara... |  5.0   |
# | Leachco Snoogle Total Body... | I have had my Snoogle for ... |  5.0   |
# | Summer Infant Complete Nur... | This Nursery and Bath Care... |  4.0   |
# | Safety 1st Tot-Lok Four Lo... | I have a wooden desk that ... |  5.0   |
# |  BABYBJORN Potty Chair - Red  | Our family is just startin... |  5.0   |
# +-------------------------------+-------------------------------+--------+


negative20 = test_data.topk('probability', k=20, reverse=True)
negative20.print_rows(num_rows=20)

# Quiz Question: Which of the following products are represented in the 20 most negative reviews? [multiple choice]
# +-------------------------------+-------------------------------+--------+
# |              name             |             review            | rating |
# +-------------------------------+-------------------------------+--------+
# | Jolly Jumper Arctic Sneak ... | I am a "research-aholic" i... |  5.0   |
# | Levana Safe N'See Digital ... | This is the first review I... |  1.0   |
# | Snuza Portable Baby Moveme... | I would have given the pro... |  1.0   |
# | Fisher-Price Ocean Wonders... | We have not had ANY luck w... |  2.0   |
# | VTech Communications Safe ... | This is my second video mo... |  1.0   |
# | Safety 1st High-Def Digita... | We bought this baby monito... |  1.0   |
# | Chicco Cortina KeyFit 30 T... | My wife and I have used th... |  1.0   |
# | Prince Lionheart Warmies W... | *****IMPORTANT UPDATE*****... |  1.0   |
# | Valco Baby Tri-mode Twin S... | I give one star to the dim... |  1.0   |
# | Adiri BPA Free Natural Nur... | I will try to write an obj... |  2.0   |
# | Munchkin Nursery Projector... | Updated January 3, 2014.  ... |  1.0   |
# | The First Years True Choic... | Note: we never installed b... |  1.0   |
# | Nuby Natural Touch Silicon... | I'm honestly confused by s... |  1.0   |
# | Peg-Perego Tatamia High Ch... | I ordered this high chair ... |  1.0   |
# |    Fisher-Price Royal Potty   | This was the worst potty e... |  1.0   |
# | Safety 1st Exchangeable Ti... | I thought it sounded great... |  1.0   |
# | Safety 1st Lift Lock and S... | Don't buy this product. If... |  1.0   |
# | Evenflo Take Me Too Premie... | I am absolutely disgusted ... |  1.0   |
# | Cloth Diaper Sprayer--styl... | I bought this sprayer out ... |  1.0   |
# | The First Years 3 Pack Bre... | I purchased several of the... |  1.0   |
# +-------------------------------+-------------------------------+--------+


# -----------------------------------
# Compute accuracy of the classifier
print "*** Compute accuracy of the classifier"
# -----------------------------------
def get_classification_accuracy(model, data, true_labels):
    # First get the predictions
    ## YOUR CODE HERE
    pred_score = model.predict(data, output_type='margin')
    #print(pred_score)
    pred_label = pred_score.apply(lambda pred_score: +1 if pred_score > 0 else -1)
    #print(pred_label)
    #print(true_labels)
    # Compute the number of correctly classified examples
    ## YOUR CODE HERE
    pred_correct = (pred_label == true_labels)
    #print(pred_correct)
    num_correct = pred_correct.sum()
    #print("num_correct = %d" % num_correct)
    #print("total = %d" % len(true_labels))
    # Then compute accuracy by dividing num_correct by total number of examples
    ## YOUR CODE HERE
    accuracy = 1.0 * num_correct / len(true_labels)
    return accuracy

print "sentiment_mode, test_data accuracy"
print get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])

# Quiz Question: What is the accuracy of the sentiment_model on the test_data? Round your answer to 2 decimal places (e.g. 0.76).
# In [163]: get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
# Out[163]: 0.9145368370530358

# Quiz Question: Does a higher accuracy value on the training_data always imply that the classifier is better?

# -----------------------------------
# Learn another classifier with fewer words
print "*** Learn another classifier with fewer words"
# -----------------------------------
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

len(significant_words)

train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)

print train_data[0]['review']

print train_data[0]['word_count']
print train_data[0]['word_count_subset']

# -----------------------------------
# Train a logistic regression model on a subset of data
print "*** Train a logistic regression model on a subset of data"
# -----------------------------------
simple_model = graphlab.logistic_classifier.create(train_data,
                                                   target = 'sentiment',
                                                   features=['word_count_subset'],
                                                   validation_set=None)
print simple_model

print "simple_model, test_data accuracy"
print get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
# Out[173]: 0.8693004559635229

# simple_model.coefficients

print simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)
# Quiz Question: Consider the coefficients of simple_model. There should be 21 of them, an intercept term + one for each word in significant_words. How many of the 20 coefficients (corresponding to the 20 significant_words and excluding the intercept term) are positive for the simple_model?
# 10


positive_coeff = simple_model.coefficients[simple_model.coefficients['value'] >= 0]
positive_coeff = positive_coeff[positive_coeff['index'] != 'None']

positive_words = positive_coeff['index']

count = len(positive_words)
for c in sentiment_model.coefficients:
    if count == 0:
        break
    if c['index'] in positive_words:
        print(c)
        count = count - 1

# Quiz Question: Are the positive words in the simple_model (let us call them positive_significant_words) also positive words in the sentiment_model?

# -----------------------------------
# Comparing models
print "*** Comparing models"
# -----------------------------------
print "sentiment_model, train_data"
print get_classification_accuracy(sentiment_model, train_data, train_data['sentiment'])
# Out[235]: 0.979440247046831

print "simple_model, train_data"
print get_classification_accuracy(simple_model, train_data, train_data['sentiment'])
# Out[236]: 0.8668150746537147

# Quiz Question: Which model (sentiment_model or simple_model) has higher accuracy on the TRAINING set?

print "sentiment_model, test_data"
print get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
# Out[237]: 0.9145368370530358

print "simple_model, test_data"
print get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
# Out[238]: 0.8693004559635229

# Quiz Question: Which model (sentiment_model or simple_model) has higher accuracy on the TEST set?

# -----------------------------------
# Baseline: Majority class prediction
print "*** Baseline: Majority class prediction"
# -----------------------------------

num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print num_positive
print num_negative

test_data['score'] = sentiment_model.predict(test_data, output_type='margin')
test_data['pred_sentiment'] = test_data['score'].apply(lambda score: +1 if score > 0 else -1)
test_data['probability'] = sentiment_model.predict(test_data, output_type='probability')

positive_sentiment = test_data[test_data['sentiment'] == 1]
positive_accuracy = len(positive_sentiment[positive_sentiment['pred_sentiment'] == 1]) / len(positive_sentiment)
print "positive_accuracy = %f" % positive_accuracy
# Out[256]: 0.9499555080975263
