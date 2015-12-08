#!/usr/bin/env python2.7

import graphlab

song_data = graphlab.SFrame('song_data.gl/')

print(song_data.head())

# --------------------
# Counting unique users
# --------------------
artists = ['Kanye West', 'Foo Fighters', 'Taylor Swift', 'Lady GaGa']
#artists = ['Kanye West']

for a in artists:
    tmp = song_data[song_data['artist'] == a]
#    print(tmp.head())
#    print(tmp['user_id'].unique().head())
    print("artist %s, uniq id %d" % (a, len(tmp['user_id'].unique())))

# --------------------
# Using groupby-aggregate to find the most popular and least popular artist
# --------------------
rank = song_data.groupby(key_columns='artist', operations={'total_count': graphlab.aggregate.SUM('listen_count')})

print(rank.head())

print("least popular: ")
print(rank.sort('total_count', ascending=True).head())
print("most popular: ")
print(rank.sort('total_count', ascending=False).head())

# --------------------
# Using groupby-aggregate to find the most recommended songs: 
# --------------------
train_data,test_data = song_data.random_split(.8,seed=0)

personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                                 user_id='user_id',
                                                                 item_id='song')

subset_test_users = test_data['user_id'].unique()[0:10000]

recommended_songs = personalized_model.recommend(subset_test_users,k=1)

print(recommended_songs.head())

rank = recommended_songs.groupby(key_columns='song', operations={'count': graphlab.aggregate.COUNT()})

print("most recommended: ")
print(rank.sort('count', ascending=False).head())
