# Francis Truong - 40087705
# COMP 472 (Artificial Intelligence) - Assignment 3
# Concordia University

# Code borrowed and adapted from https://www.kdnuggets.com/2020/07/spam-filter-python-naive-bayes-scratch.html

import pandas as pd
import re
import numpy as np
import csv


training = pd.read_csv('covid_training.tsv', sep='\t', header=None, skiprows=[0], names=['tweet_id', 'text', 'q1_label',
                                                                                         'q2_label', 'q3_label',
                                                                                         'q4_label',
                                                                                         'q5_label', 'q6_label',
                                                                                         'q7_label'])
test = pd.read_csv('covid_test_public.tsv', sep='\t', header=None, names=['tweet_id', 'text', 'q1_label',
                                                                          'q2_label', 'q3_label', 'q4_label',
                                                                          'q5_label', 'q6_label', 'q7_label'])
# print(training.shape)
# print(training.head())

print(training['q1_label'].value_counts(normalize=True))

training['text'] = training['text'].str.replace('\W', ' ')  # removes punctuation
training['text'] = training['text'].str.lower()  # turn letters into lowercase
training['text'] = training['text'].str.split()  # split sentences into individual words


def remove_singles(l):
    once = set()
    more = set()
    for w in l:
        if w not in more:
            if w in once:
                more.add(w)
                once.remove(w)
            else:
                once.add(w)
    return more


vocab = []
for tweet in training['text']:
    for word in tweet:
        vocab.append(word)

filtered_vocab = remove_singles(vocab)
vocab = list(set(vocab))  # removes duplicates in vocab
filtered_vocab = list(set(filtered_vocab))

print(vocab)
print(len(vocab))
print(filtered_vocab)
print(len(filtered_vocab))

tweet_word_count = {unique_word: [0] * len(training['text']) for unique_word in vocab}
for index, tweet in enumerate(training['text']):
    for word in tweet:
        tweet_word_count[word][index] += 1

word_counts = pd.DataFrame(tweet_word_count)
print(word_counts.head())

training_clean = pd.concat([training, word_counts], axis=1)
print(training_clean.head())

yes_tweets = training_clean[training_clean['q1_label'] == 'yes']
no_tweets = training_clean[training_clean['q1_label'] == 'no']

p_yes = len(yes_tweets) / len(training_clean)
p_no = len(no_tweets) / len(training_clean)

print(p_yes)
print(p_no)

n_words_per_yes_tweets = yes_tweets['text'].apply(len)
n_yes = n_words_per_yes_tweets.sum()

print(n_yes)

n_words_per_no_tweets = no_tweets['text'].apply(len)
n_no = n_words_per_no_tweets.sum()

print(n_no)

n_vocab = len(vocab)

smoothing = 0.01

parameters_yes = {unique_word: 0 for unique_word in vocab}
parameters_no = {unique_word: 0 for unique_word in vocab}

for word in vocab:
    n_word_given_yes = yes_tweets[word].sum()
    p_word_given_yes = (n_word_given_yes + smoothing) / (n_yes + smoothing * n_vocab)
    parameters_yes[word] = p_word_given_yes

    n_word_given_no = no_tweets[word].sum()  # spam_messages already defined
    p_word_given_no = (n_word_given_no + smoothing) / (n_no + smoothing * n_vocab)
    parameters_no[word] = p_word_given_no


def classify(tweet):
    '''
   message: a string
   '''

    tweet = re.sub('\W', ' ', tweet)
    tweet = tweet.lower().split()

    #print(tweet)

    p_y_tweet = p_yes
    p_no_tweet = p_no

    for word in tweet:
        if word in parameters_yes:
            p_y_tweet *= parameters_yes[word]

        if word in parameters_no:
            p_no_tweet *= parameters_no[word]

    p_y_tweet = np.log10(p_y_tweet)
    p_no_tweet = np.log10(p_no_tweet)
    p_y_tweet = round(p_y_tweet, 2)
    p_no_tweet = round(p_no_tweet, 2)

    #print('P(yes|tweet):', p_y_tweet)
    #print('P(no|tweet):', p_no_tweet)

    if p_y_tweet > p_no_tweet:
       return('yes', str(p_y_tweet))
        #print('Label: yes')
    elif p_no_tweet > p_y_tweet:
       return('no', str(p_no_tweet))
        #print('Label: no')


test_result = {}

with open("covid_test_public.tsv", encoding='utf8') as f:
    test = csv.reader(f, delimiter='\t')
    count = 0
    for line in test:
        id = line[0]
        tweet = str(line[1])
        target = line[2]
        result, score = classify(tweet)
        if target == result:
           label = 'correct'
        else:
           label = 'wrong'
        output = id + "  " + result + "  " + score + "  " + target + "  " + label
        test_result[count] = output
        count += 1

print(test_result)
