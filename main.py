#
# VINCENT CERRI
# 40034135
# COMP 472 - Artificial Intelligence
# Sunday, April 5th, 2020
#
import math

import numpy as np
import sys
from nltk import ngrams, everygrams, pprint
import nltk
from nltk.probability import FreqDist
from _collections import defaultdict

def determine_language(tweet):
    lowest_probability = 0.0  # 1.0 is the highest possible probability so we will update this value with the calculated results
    language = ""
    for model in models:
        new_probability = determine_probability(model, tweet)

        if (new_probability > lowest_probability):
            lowest_probability = new_probability
            language = model
    return language

def determine_probability(model, tweet):
    probability = math.log((vocab_portion[model] / total_vocab), 10)
    tweet_bigram = list(ngrams(tweet.lower(), 2))
    #print(model)
    #print(models[model])

    for gram in tweet_bigram:
        if not (gram[0] not in vocab_list or gram[1] not in vocab_list):
            # what if the gram is not in the model? This is when we will need to be using smoothing.
            if gram not in models[model]:
                print ("Gram not in model. Needs smoothing")
                # this is temporary until we add smoothing
            else:
                #probability = probability * models[model][gram]   # multiply by the proability of getting the 2 character string
                probability = probability + math.log(models[model][gram], 10)  # using the addition of log base 10 for the probability
    #print(model, " : log probability = ", probability)
    #print(model, " : actual probability = ", math.exp(probability))
    # note to self.
    # log(P(A,B,C)) = log(P(A)) + log(P(B)) + log(P(C))
    # P(A,B,C) = e ^ log(P(A,B,C))
    return math.exp(probability)


if __name__ == "__main__":

    vocabulary = 0
    # 0 -> Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]
    # 1 -> Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
    # 2 -> Distinguish up and low cases and use all characters accepted by the built-in isalpha() method
    n_gram_size = 2  # n-gram indicates into how many characters we should be splitting up the strings
    # 1 -> character unigrams
    # 2 -> character bigrams
    # 3 -> character trigrams
    smoothing_value = 1  # a real in the interval [0...1]
    train_filename = ""
    test_filename = ""

    # models is a dictionary of all the possible models that we will be using
    models = dict()
    ca_model = dict()
    eu_model = dict()
    es_model = dict()
    en_model = dict()
    gl_model = dict()
    pt_model = dict()
    models['ca'] = ca_model
    models['eu'] = eu_model
    models['es'] = es_model
    models['en'] = en_model
    models['gl'] = gl_model
    models['pt'] = pt_model
    
    # this tracks the amount of training lines read, and how many per language
    total_vocab = 0
    vocab_portion = { 'ca':0, 'eu':0, 'es':0, 'en':0, 'gl':0, 'pt':0 }

    if vocabulary == 0:
        vocab_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                      't', 'u', 'v', 'w', 'x', 'y', 'z']
    elif vocabulary == 1:
        vocab_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                      't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                      'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # this adds delta smoothing
    # we want to create the frequency array for each model depending on if we are using 1,2 or 3 character ngrams
    for model in models:
        if (n_gram_size == 1):
            for i in vocab_list:
                models[model][i] = smoothing_value                
        elif (n_gram_size == 2):            
            for i in vocab_list:
                for j in vocab_list:
                    models[model][(i, j)] = smoothing_value
        elif (n_gram_size == 3):
            for i in vocab_list:
                for j in vocab_list:
                    for k in vocab_list:
                        models[model][(i, j, k)] = smoothing_value
        print(models[model])

    # we will start by reading from the training tweets file to create our models
    f = open("training-tweets.txt", "r", encoding="utf8")
    fileInput = f.readline()

    # this is the loop that reads the file input line by line
    while fileInput:
        #print(fileInput)
        language = fileInput.split("\t")[2]
        string_to_add = fileInput.split("\t")[3]
        vocab_portion[language] += 1
        total_vocab += 1
        print(string_to_add)

        my_ngram = list(ngrams(string_to_add.lower(), n_gram_size))
        for gram in my_ngram:
            # we dont want ' " included in our dictionary as it is not part of the vocabulary
            is_gram_in_vocab = True
            for i in range(n_gram_size):  # loop from 0 to n_gram_size - 1
                if (gram[i] not in vocab_list):
                    is_gram_in_vocab = False
            if is_gram_in_vocab:
                if language not in models.keys():
                    print("sorry that language doesn't have a model")
                else:
                    models[language][gram] += 1

        fileInput = f.readline()
    f.close()

    # lets convert each of the models to a probability instead of a count. We should also apply the smoothing value here
    for language, dictionary in models.items():
        nmb_items = 0
        print("\nModel langauge:", language)
        print(dictionary)

        for key in dictionary.keys():
            nmb_items += dictionary[key]
        for k in dictionary.keys():
            dictionary[k] = dictionary[k] / float(nmb_items)

    for key, dictionary in models.items():
        print(key, " : ", dictionary)



    # to avoid arithmetic underflow work in log10 space
    # this means that instead of doing the product of probabilities, we instead add the log of the probabilities


    # now that we have models with the probabilities, we want to look at the test set to see if we can determine the
    # language of each tweet.
    # read from the test files and try determining the language
    f = open("test-tweets-given.txt", "r", encoding="utf-8")
    fileInput = f.readline()

    solution_file = open("trace_myModel", "w", encoding="utf-8")  # this is the solution file that we will write to

    total_lines = 0
    total_correct = 0

    # this is the loop that reads the file input line by line
    while fileInput:
        # print(fileInput)
        tweet_id = fileInput.split("\t")[0]
        tweet_author = fileInput.split("\t")[1]
        tweet_language = fileInput.split("\t")[2]
        tweet = fileInput.split("\t")[3]

        language = determine_language(tweet)
        print(language)

        print(tweet)

        solution_file.write((tweet_id + "\tactual language: " + tweet_language + "\tpredicted language:" + language + "\n"))

        if tweet_language == language:
            total_correct += 1
        total_lines += 1

        fileInput = f.readline()

    total_percentage = float(total_correct / total_lines) * 100
    solution_file.write(("total lines: " + str(total_lines) + "\ttotal correct: " + str(total_correct) + "\tpercentage correct: " + str(total_percentage) + "%\n"))
    f.close()