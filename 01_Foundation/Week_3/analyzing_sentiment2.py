#!/usr/local/bin/python

import graphlab

products = graphlab.SFrame('amazon_baby.gl/')

products['word_count'] = graphlab.text_analytics.count_words(products['review'])

print(products.head())

# -------------------------------------------
# selected word model
# -------------------------------------------
# 1. Use .apply() to build a new feature with the counts for each of the selected_words:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

def count_word(word_count, word):
    if word in word_count:
        return word_count[word]
    else:
        return 0

def awesome_count(word_count):
    return count_word(word_count, 'awesome')

def great_count(word_count):
    return count_word(word_count, 'great')

def fantastic_count(word_count):
    return count_word(word_count, 'fantastic')

def amazing_count(word_count):
    return count_word(word_count, 'amazing')

def love_count(word_count):
    return count_word(word_count, 'love')

def horrible_count(word_count):
    return count_word(word_count, 'horrible')

def bad_count(word_count):
    return count_word(word_count, 'bad')

def terrible_count(word_count):
    return count_word(word_count, 'terrible')

def awful_count(word_count):
    return count_word(word_count, 'awful')

def wow_count(word_count):
    return count_word(word_count, 'wow')

def hate_count(word_count):
    return count_word(word_count, 'hate')

for w in selected_words:
    if w == 'awesome':
        products[w] = products['word_count'].apply(awesome_count)
    elif w == 'great':
        products[w] = products['word_count'].apply(great_count)
    elif w == 'fantastic':
        products[w] = products['word_count'].apply(fantastic_count)
    elif w == 'amazing':
        products[w] = products['word_count'].apply(amazing_count)
    elif w == 'love':
        products[w] = products['word_count'].apply(love_count)
    elif w == 'horrible':
        products[w] = products['word_count'].apply(horrible_count)
    elif w == 'bad':
        products[w] = products['word_count'].apply(bad_count)
    elif w == 'terrible':
        products[w] = products['word_count'].apply(terrible_count)
    elif w == 'awful':
        products[w] = products['word_count'].apply(awful_count)
    elif w == 'wow':
        products[w] = products['word_count'].apply(wow_count)
    elif w == 'hate':
        products[w] = products['word_count'].apply(hate_count)

print(products.head())

wc = {}
for w in selected_words:
    wc[w] = sum(products[w])

print(wc)

# awesome: 2090
# great: 45206
# fantastic: 932
# amazing: 1363
# love: 42065
# horrible: 734
# bad: 3724
# terrible: 748
# awful: 383
# wow: 144
# hate: 1220

# 2. Create a new sentiment analysis model using only the selected_words as features
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'] >=4

train_data,test_data = products.random_split(.8, seed=0)

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                           target='sentiment',
                                                           features=selected_words,
                                                           validation_set=test_data)

print(selected_words_model['coefficients'])

selected_words_model['coefficients'].sort('value', ascending=False)

selected_words_model['coefficients'].sort('value', ascending=True)

# Most positive: love, 1.39989834302
# Most nevative: terrible, -2.09049998487

# 3. Comparing the accuracy of different sentiment analysis model

selected_words_model.evaluate(test_data)

#  +--------------+-----------------+-------+
#  | target_label | predicted_label | count |
#  +--------------+-----------------+-------+
#  |      0       |        0        |  234  |
#  |      1       |        0        |  130  |
#  |      0       |        1        |  5094 |
#  |      1       |        1        | 27846 |
#  +--------------+-----------------+-------+

# the accuracy of the selected_words_model: 0.8431419649291376
# the accuracy of the sentiment_model: 0.916256305549

# -------------------------------------------
# diaper_champ_reviews with the sentiment model
# -------------------------------------------
diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']
len(diaper_champ_reviews)

print(products.head())

train_data,test_data = products.random_split(.8, seed=0)

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)

print(sentiment_model.evaluate(test_data))

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')

print(diaper_champ_reviews.sort('predicted_sentiment', ascending=False))

diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)

print(selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability'))
