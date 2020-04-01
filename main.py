#
# VINCENT CERRI
# 40034135
# COMP 472 - Artificial Intelligence
# Sunday, April 5th, 2020
#

import numpy as np
import sys
from nltk import ngrams, everygrams, pprint
import nltk
from nltk.probability import FreqDist
from _collections import defaultdict
#nltk.download('punkt')

#def run_report(self, vocab, ngram_size, smooth_value):
    #if vocab == 0:
    #    # 0 -> Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]
    #elif vocab == 1:
    #    # 1 -> Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
    #elif vocab == 2:
    #    # 2 -> Distinguish up and low cases and use all characters accepted by the built-in isalpha() method

if __name__ == "__main__":

    vocabulary = 0
    # 0 -> Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]
    # 1 -> Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
    # 2 -> Distinguish up and low cases and use all characters accepted by the built-in isalpha() method
    n_gram_size = 0
    # 1 -> character unigrams
    # 2 -> character bigrams
    # 3 -> character trigrams
    smoothing_value = 0  # a real in the interval [0...1]
    train_filename = ""
    test_filename = ""

    # dictionary list is a list of all the possible models that we will be using
    models = dict()
    ca_model = dict()
    eu_model = dict()
    es_model = dict()
    en_model = dict()
    pt_model = dict()
    models['ca'] = ca_model
    models['eu'] = eu_model
    models['es'] = es_model
    models['en'] = en_model
    models['pt'] = pt_model

    if vocabulary == 0:
        vocab_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                      't', 'u', 'v', 'w', 'x', 'y', 'z']

    f = open("C:\\Users\\vince\\OneDrive - Concordia University - Canada\\CONCORDIA\\Winter 2020\\COMP 472\\Project2\\training-tweets.txt", "r", encoding="utf8")
    fileInput = f.readline()

    # this is the loop that reads the file input line by line
    while fileInput:
        #print(fileInput)
        language = fileInput.split("\t")[2]
        string_to_add = fileInput.split("\t")[3]
        print(string_to_add)

        my_bigrams = list(ngrams(string_to_add, 2))

        for gram in my_bigrams:
            # we dont want " " included in our dictionary as it is not part of the vocabulary
            if not (gram[0] not in vocab_list or gram[1] not in vocab_list):
                if language not in models.keys():
                    print ("sorry that languge doesn't have a modle")
                else:
                    if gram not in models[language].keys():
                        models[language][gram] = 1
                    else:
                        models[language][gram] += 1
        fileInput = f.readline()
    f.close()


    # lets convert each of the models to a probability instead of a count
    for language, dictionary in models.items():
        nmb_items = 0
        print("\nModel langauge:", language)
        print(dictionary)

        for key in dictionary.keys():
            nmb_items += dictionary[key]
        for k in dictionary.keys():
            dictionary[k] = dictionary[k] / float (nmb_items)

    for key, dictionary in models.items():
        print(key, " : ", dictionary)










    sentence = "this is a foo bar sentences and i want to ngramize it"
    n = 3
    #bigrams = ngrams(sentence.split(), n)

    #list(everygrams(sentence, 2, 2))

    #for grams in list(everygrams(sentence, 2, 2)):
    #    print(grams)

    #for grams in bigrams:
    #    print(grams)

    #y = []
    #for x in range(len(sentence) - n + 1):
    #    y.append(sentence[x:x+n])
    #print(y)


    # must apply the chain rule when calculating the probability
    # how do we have the probabilities? Create a dictionary of terms and their probabilities based on the data set.


    my_bigrams = list(ngrams(sentence, 2))
    #my_trigrams = list(ngrams(sentence, 3))

    #print (my_bigrams)
    #print (my_trigrams)

    # create an empy dictionary to store the bigram model
    # the format will be [ ('a','b') , count ]
    model = dict()
    nmb_items = 0
    total_prob = 0.0

    for gram in my_bigrams:
        # we dont want " " included in our dictionary as it is not part of the vocabulary
        if not (gram[0] not in vocab_list or gram[1] not in vocab_list):
            if gram not in model.keys():
                model[gram] = 1
            else:
                model[gram] += 1
            nmb_items += 1

    print("Number of items: ", nmb_items)

    # convert from count to a probability
    for k in model.keys():
        model[k] = model[k] / float(nmb_items)

    print(sorted(((v, k) for k, v in model.items()), reverse=True))  # to print out the sorted dictionary based on probabilities

    for k in model.keys():
        total_prob += model[k]
    print(total_prob)






