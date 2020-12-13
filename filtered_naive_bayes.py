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
# print(training['q1_label'].value_counts(normalize=True))

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
original_vocab = list(set(vocab))  # removes duplicates in vocab
vocab = list(set(filtered_vocab))

# print(vocab)
# print(len(original_vocab))
# print(len(filtered_vocab))

tweet_word_count = {unique_word: [0] * len(training['text']) for unique_word in vocab}
for index, tweet in enumerate(training['text']):
    for word in tweet:
        if word in vocab:
            tweet_word_count[word][index] += 1

word_counts = pd.DataFrame(tweet_word_count)

training_clean = pd.concat([training, word_counts], axis=1)

yes_tweets = training_clean[training_clean['q1_label'] == 'yes']
no_tweets = training_clean[training_clean['q1_label'] == 'no']

p_yes = len(yes_tweets) / len(training_clean)
p_no = len(no_tweets) / len(training_clean)

n_words_per_yes_tweets = yes_tweets['text'].apply(len)
n_yes = n_words_per_yes_tweets.sum()

n_words_per_no_tweets = no_tweets['text'].apply(len)
n_no = n_words_per_no_tweets.sum()

n_vocab = len(vocab)

smoothing = 0.01

parameters_yes = {unique_word: 0 for unique_word in vocab}
parameters_no = {unique_word: 0 for unique_word in vocab}

for word in vocab:
    n_word_given_yes = yes_tweets[word].sum()
    p_word_given_yes = (n_word_given_yes + smoothing) / (n_yes + smoothing * n_vocab)
    parameters_yes[word] = p_word_given_yes

    n_word_given_no = no_tweets[word].sum()
    p_word_given_no = (n_word_given_no + smoothing) / (n_no + smoothing * n_vocab)
    parameters_no[word] = p_word_given_no


def classify(tweet):
    '''
   message: a string
   '''

    tweet = re.sub('\W', ' ', tweet)
    tweet = tweet.lower().split()

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

    if p_y_tweet > p_no_tweet:
        return 'yes', str(p_y_tweet)
    elif p_no_tweet > p_y_tweet:
        return 'no', str(p_no_tweet)


test_result = {}

n_correct = 0
n_wrong = 0
yes_tp = 0
no_tp = 0
false_positive = 0
false_negative = 0

with open("covid_test_public.tsv", encoding='utf8') as f:
    test = csv.reader(f, delimiter='\t')
    count = 0
    for line in test:
        tweet_id = line[0]
        tweet = str(line[1])
        target = line[2]
        result, score = classify(tweet)
        if target == 'yes' and result == 'no':
            false_negative += 1
        if target == 'no' and result == 'yes':
            false_positive += 1
        if target == 'yes' and result == 'yes':
            yes_tp += 1
        if target == 'no' and result == 'no':
            no_tp += 1
        if target == result:
            label = 'correct'
            n_correct += 1
        else:
            label = 'wrong'
            n_wrong += 1
        output = tweet_id + "  " + result + "  " + score + "  " + target + "  " + label
        test_result[count] = output
        count += 1

print(test_result)

with open('./trace_NB-BOW-FV.txt', 'w') as f:
    for content in test_result.values():
        f.write(content)
        f.write('\n')
    f.close()

accuracy = round(n_correct / (n_correct + n_wrong), 4)
yes_precision = round(yes_tp / (yes_tp + false_positive), 4)
no_precision = round(no_tp / (no_tp + false_positive), 4)
yes_recall = round(yes_tp / (yes_tp + false_negative), 4)
no_recall = round(no_tp / (no_tp + false_negative), 4)
yes_f1 = round((2 * yes_precision * yes_recall) / (yes_precision + yes_recall), 4)
no_f1 = round((2 * no_precision * no_recall) / (no_precision + no_recall), 4)

with open('./eval_NB-BOW-FV.txt', 'w') as f:
    f.write(str(accuracy) + '\n')
    f.write(str(yes_precision) + "  " + str(no_precision) + '\n')
    f.write(str(yes_recall) + "  " + str(no_recall) + '\n')
    f.write(str(yes_f1) + "  " + str(no_f1) + '\n')
    f.close()
