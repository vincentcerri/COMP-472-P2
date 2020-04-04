#
# VINCENT CERRI
# 40034135
# COMP 472 - Artificial Intelligence
# Sunday, April 5th, 2020
#
import math
from decimal import Decimal

import numpy as np
import sys
from nltk import ngrams, everygrams, pprint
import nltk
from nltk.probability import FreqDist
from _collections import defaultdict


class NaturalLanguageProcessing:
    def __init__(self):
        self.vocabulary = 0
        # 0 -> Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]
        # 1 -> Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
        # 2 -> Distinguish up and low cases and use all characters accepted by the built-in isalpha() method
        self.n_gram_size = 1  # n-gram indicates into how many characters we should be splitting up the strings
        # 1 -> character unigrams
        # 2 -> character bigrams
        # 3 -> character trigrams
        self.smoothing_value = 1.0  # a real in the interval [0...1]
        self.models = dict()  # models is a dictionary of all the possible models that we will be using
        self.ca_model = dict()
        self.eu_model = dict()
        self.es_model = dict()
        self.en_model = dict()
        self.gl_model = dict()
        self.pt_model = dict()
        self.models['ca'] = self.ca_model
        self.models['eu'] = self.eu_model
        self.models['es'] = self.es_model
        self.models['en'] = self.en_model
        self.models['gl'] = self.gl_model
        self.models['pt'] = self.pt_model
        # this tracks the amount of training lines read, and how many per language
        self.vocab_list = []
        self.vocab_list_size = 0
        self.total_vocab = 0
        self.vocab_portion = {'ca': 0, 'eu': 0, 'es': 0, 'en': 0, 'gl': 0, 'pt': 0}
        # this tracks the amount of training ngrams per language
        self.ngram_portion = {'ca': 0, 'eu': 0, 'es': 0, 'en': 0, 'gl': 0, 'pt': 0}

        self.tp_count = {'ca': 0, 'eu': 0, 'es': 0, 'en': 0, 'gl': 0, 'pt': 0}
        self.fp_count = {'ca': 0, 'eu': 0, 'es': 0, 'en': 0, 'gl': 0, 'pt': 0}
        self.trace_probability = 0.0

    def produce_output(self, vocabulary, n_gram_size, smoothing_value, train_name, test_name):
        self.vocabulary = vocabulary
        self.n_gram_size = n_gram_size
        self.smoothing_value = smoothing_value

        if vocabulary == 0:
            self.vocab_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                               's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            self.vocab_list_size = len(self.vocab_list)
        elif vocabulary == 1:
            self.vocab_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                               's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            self.vocab_list_size = len(self.vocab_list)
        elif vocabulary == 2:
            self.vocab_list = []
            self.vocab_list_size = len(self.vocab_list)

        # we will start by reading from the training tweets file to create our models
        f = open(train_name, "r", encoding="utf8")
        fileInput = f.readline()

        # this is the loop that reads the file input line by line
        while fileInput:
            # print(fileInput)
            language = fileInput.split("\t")[2]
            string_to_add = fileInput.split("\t")[3]
            self.vocab_portion[language] += 1
            self.total_vocab += 1
            print(string_to_add)

            if vocabulary == 0:
                my_ngram = list(ngrams(string_to_add.lower(), n_gram_size))  # convert the tweet to lower case
            else:
                my_ngram = list(ngrams(string_to_add, n_gram_size))  # leave the tweet as is

            for gram in my_ngram:
                # we dont want ' ' included in our dictionary as it is not part of the vocabulary
                is_gram_in_vocab = True
                if vocabulary == 2:
                    for i in range(n_gram_size):  # loop from 0 to n_gram_size - 1 to determine if the gram is in the vocabulary
                        if not gram[i].isalpha():
                            is_gram_in_vocab = False  # if the gram is not isalpha() then we shouldn't add it to the list
                    if is_gram_in_vocab:
                        if gram not in self.vocab_list:  # if the gram is not in the vocab list BUT it isalpha() we want to add it in
                            self.vocab_list_size += 1
                            self.vocab_list.append(gram)
                else:  # if vocab == 0 or 1
                    for i in range(n_gram_size):  # loop from 0 to n_gram_size - 1 to determine if the gram is in the vocabulary
                        if gram[i] not in self.vocab_list:
                            is_gram_in_vocab = False
                            # if vocab = 0, 1 or 2
                if is_gram_in_vocab:
                    if language not in self.models.keys():
                        print("sorry that language doesn't have a model")
                    elif gram not in self.models[language].keys():
                        self.models[language][gram] = 1 + smoothing_value
                        self.ngram_portion[language] += 1
                    else:
                        self.models[language][gram] += 1
                        self.ngram_portion[language] += 1

            fileInput = f.readline()
        f.close()

        # lets convert each of the models to a probability instead of a count. We should also apply the smoothing value here
        for language, dictionary in self.models.items():
            nmb_items = 0
            print("\nModel langauge:", language)
            print(dictionary)

            for k in dictionary.keys():
                # no need to add the smoothing value to the numerator as it is already implemented when counting
                dictionary[k] = (dictionary[k]) / float(self.ngram_portion[language] + smoothing_value * (math.pow(self.vocab_list_size, n_gram_size)))

        # to avoid arithmetic underflow work in log10 space
        # this means that instead of doing the product of probabilities, we instead add the log of the probabilities

        # now that we have models with the probabilities, we want to look at the test set to see if we can determine the
        # language of each tweet.
        # read from the test files and try determining the language
        f = open(test_name, "r", encoding="utf-8")
        fileInput = f.readline()

        trace_filename = str("trace_" + str(vocabulary) + "_" + str(n_gram_size) + "_" + str(smoothing_value))
        solution_trace_file = open(trace_filename, "w", encoding="utf-8")  # this is the trace file that the program will write to
        overall_eval_filename = str("eval_" + str(vocabulary) + "_" + str(n_gram_size) + "_" + str(smoothing_value))
        overall_eval_file = open(overall_eval_filename, "w", encoding="utf-8")  # this is the eval file that the program will write to
        total_lines = 0
        total_correct = 0

        # this is the loop that reads the file input line by line
        while fileInput:
            # print(fileInput)
            tweet_id = fileInput.split("\t")[0]
            tweet_author = fileInput.split("\t")[1]
            tweet_language = fileInput.split("\t")[2]
            tweet = fileInput.split("\t")[3]

            language = self.determine_language(tweet)

            if tweet_language == language:
                total_correct += 1
                language_match = "correct"
                self.tp_count[language] += 1
            else:
                language_match = "wrong"
                self.fp_count[language] += 1

            total_lines += 1

            solution_trace_file.write((tweet_id + "  " + language + "  " + str("{:.2E}".format(Decimal(self.trace_probability))) + "  " + tweet_language + "  " + language_match + "\n"))

            fileInput = f.readline()

        # calculate the accuracy
        accuracy = float(total_correct / total_lines)
        overall_eval_file.write(str('{:.4f}'.format(accuracy)) + "\n")
        # calculate the precision
        # precision = TP / (TP + FP)
        # TP: Your system designated a tweet a certain language. The tweet is in fact that language.
        # FP: Your system designated a tweet a certain langauge. The tweet is not actually that language. The system is wrong
        print(self.vocab_portion)
        for lang in self.models:
            if self.tp_count[lang] == 0:
                precision = 0
            else:
                precision = self.tp_count[lang] / (self.tp_count[lang] + self.fp_count[lang])
            overall_eval_file.write(str('{:.4f}'.format(precision)) + "  ")
        overall_eval_file.write("\n")
        # calculate the recall
        # recall = TP / (TP + FN)
        # FN: Your system designated a tweet as not a certain language. The tweet was in fact that language
        # IM NOT SURE WHAT THE DIFFERENCE IS BETWEEN FN AND FP

        # calcualte the F1 measure

        # calcualte the macro-F1 and weighed-average-F1


        solution_trace_file.close()
        overall_eval_file.close()
        f.close()

    def determine_language(self, tweet):
        best_probability = 0.0  # 1.0 is the highest possible probability so we will update this value with the calculated results
        language = ""
        for model in self.models:
            new_probability = self.determine_probability(model, tweet)

            if new_probability > best_probability:
                best_probability = new_probability
                language = model
        return language

    def determine_probability(self, model, tweet):
        probability = math.log((self.vocab_portion[model] / self.total_vocab), 10)
        if self.vocabulary == 0:
            tweet_gram = list(ngrams(tweet.lower(), self.n_gram_size))  # convert the tweet to lower case
        else:
            tweet_gram = list(ngrams(tweet, self.n_gram_size))  # leave the tweet as is

        # calculating probability for non existant grams (i.e. smoothing value only) outside of for loop for lessened computation load
        smooth_probability_log = math.log((self.smoothing_value/float(self.ngram_portion[model] + self.smoothing_value*(math.pow(self.vocab_list_size, self.n_gram_size)))), 10)

        for gram in tweet_gram:
            is_gram_in_vocab = True
            for j in range(self.n_gram_size):  # first determine if the gram is in the vocabulary. If not, we wont consider it
                if gram[j] not in self.vocab_list:
                    is_gram_in_vocab = False
            if is_gram_in_vocab:  # if the gram is actually in the vocabulary
                if gram not in self.models[model]:  # if the gram is not in the model
                    probability = probability + smooth_probability_log
                else:  # if the gram is in the model
                    probability = probability + math.log(self.models[model][gram], 10)  # using the addition of log base 10 for the probability

        # note to self.
        # log(P(A,B,C)) = log(P(A)) + log(P(B)) + log(P(C))
        # P(A,B,C) = e ^ log(P(A,B,C))
        self.trace_probability = probability
        return math.exp(probability)

if __name__ == "__main__":
    nlp = NaturalLanguageProcessing()
    nlp.produce_output(0, 1, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(1, 1, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(2, 1, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(0, 2, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(1, 2, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(2, 2, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(0, 3, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(1, 3, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(2, 3, 0.5, "training-tweets.txt", "test-tweets-given.txt")

    print("WE ARE DONE WITH THE TESTS")

