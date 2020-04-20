#
# VINCENT CERRI - 40034135
# SEAN HOWARD - 26346685
# COMP 472 - Artificial Intelligence
# Sunday, April 5th, 2020
#
import math
import re
from decimal import Decimal
from nltk import ngrams, everygrams, pprint


class NaturalLanguageProcessing:
    def __init__(self):
        self.vocabulary = 0
        # 0 -> Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]
        # 1 -> Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]
        # 2 -> Distinguish up and low cases and use all characters accepted by the built-in isalpha() method
        # 3 -> Our custom BYOM
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
        # This is the probability for non existant grams (i.e. smoothing value only), in log10 space
        self.smooth_probability_log = {'ca': 0, 'eu': 0, 'es': 0, 'en': 0, 'gl': 0, 'pt': 0}

        self.tp_count = {'ca': 0, 'eu': 0, 'es': 0, 'en': 0, 'gl': 0, 'pt': 0}
        self.fp_count = {'ca': 0, 'eu': 0, 'es': 0, 'en': 0, 'gl': 0, 'pt': 0}
        self.actual_language_count = {'ca': 0, 'eu': 0, 'es': 0, 'en': 0, 'gl': 0, 'pt': 0}
        self.precision_values = {'ca': 0.0, 'eu': 0.0, 'es': 0.0, 'en': 0.0, 'gl': 0.0, 'pt': 0.0}
        self.recall_values = {'ca': 0.0, 'eu': 0.0, 'es': 0.0, 'en': 0.0, 'gl': 0.0, 'pt': 0.0}
        self.f1_values = {'ca': 0.0, 'eu': 0.0, 'es': 0.0, 'en': 0.0, 'gl': 0.0, 'pt': 0.0}
        self.trace_probability = 0.0

    def produce_output(self, vocabulary, n_gram_size, entered_smoothing_value, train_name, test_name):
        self.vocabulary = vocabulary
        self.n_gram_size = n_gram_size
        if (entered_smoothing_value > 1):
            smoothing_value = 1     # encase someone enters too big a number
        elif (entered_smoothing_value < 0):
            smoothing_value = 0     # encase someone enters a negative number
        else:
            smoothing_value = entered_smoothing_value
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
            self.vocab_list = [ ]   # need to make sure it is blank encase another vocabulary was run prior
            self.vocab_list_size = 116766   # all possible characters from isalpha()
        elif vocabulary == 3:  # vocab == 3 is our BYOM vocabulary
            self.vocab_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                               's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                               'À', 'Á', 'Â', 'Ã', 'Ç', 'É', 'Ê', 'Ó', 'Ñ', 'Õ', 'Ü', 'Í', 'Ì', 'Ï', 'Ú', 
                               'à', 'á', 'â', 'ã', 'ç', 'é', 'ê', 'ó', 'õ', 'ú', 'ü', 'ñ', 'í', 'ï', 'ì']
            self.vocab_list_size = len(self.vocab_list)

        # we will start by reading from the training tweets file to create our models
        f = open(train_name, "r", encoding="utf8")
        fileInput = f.readline()

        ca_test_language_count = 0
        eu_test_language_count = 0
        es_test_language_count = 0
        en_test_language_count = 0
        gl_test_language_count = 0
        pt_test_language_count = 0


        # this is the loop that reads the file input line by line
        while fileInput:
            language = fileInput.split("\t")[2]
            string_to_add = fileInput.split("\t")[3]
            self.vocab_portion[language] += 1
            self.total_vocab += 1

            if language == "eu":
                eu_test_language_count += 1
            elif language == "ca":
                ca_test_language_count += 1
            elif language == "es":
                es_test_language_count += 1
            elif language == "en":
                en_test_language_count += 1
            elif language == "gl":
                gl_test_language_count += 1
            elif language == "pt":
                pt_test_language_count += 1

            if vocabulary == 0:
                my_ngram = list(ngrams(string_to_add.lower(), n_gram_size))  # convert the tweet to lower case
            elif vocabulary == 3:
                string_to_add = self.clean_string(string_to_add)
                my_ngram = list(ngrams(string_to_add, n_gram_size)) 
            else:
                my_ngram = list(ngrams(string_to_add, n_gram_size))  # leave the tweet as is

            for gram in my_ngram:
                # we dont want ' ' included in our dictionary as it is not part of the vocabulary
                is_gram_in_vocab = True
                if vocabulary == 2:
                    for i in range(n_gram_size):  # loop from 0 to n_gram_size - 1 to determine if the gram is in the vocabulary
                        if not gram[i].isalpha():
                            is_gram_in_vocab = False
                else:   # if vocab == 0 or 1 or 3
                    for i in range(n_gram_size):  # loop from 0 to n_gram_size - 1 to determine if the gram is in the vocabulary
                        if gram[i] not in self.vocab_list:
                            is_gram_in_vocab = False
                            break
                if is_gram_in_vocab:
                    if language not in self.models.keys():
                        print("sorry that language doesn't have a model")
                    elif gram not in self.models[language].keys():
                        self.models[language][gram] = 1 + smoothing_value
                        self.ngram_portion[language] += 1  # tracks the amount of training ngrams
                    else:
                        self.models[language][gram] += 1
                        self.ngram_portion[language] += 1

            fileInput = f.readline()
        f.close()

        print("ca: " + str(ca_test_language_count))
        print("eu: " + str(eu_test_language_count))
        print("es: " + str(es_test_language_count))
        print("en: " + str(en_test_language_count))
        print("gl: " + str(gl_test_language_count))
        print("pt: " + str(pt_test_language_count))



        # lets convert each of the models to a probability instead of a count. We should also apply the smoothing value here
        for language, dictionary in self.models.items():
            #print("\nModel langauge:", language)
            #print(dictionary)

            for k in dictionary.keys():
                # no need to add the smoothing value to the numerator as it is already implemented when counting
                dictionary[k] = (dictionary[k]) / float(self.ngram_portion[language] + smoothing_value * (math.pow(self.vocab_list_size, n_gram_size)))

        # to avoid arithmetic underflow work in log10 space
        # this means that instead of doing the product of probabilities, we instead add the log of the probabilities

        # calculating smooth_probability_log now that all its variables exist
        if (smoothing_value > 0):   # if smoothing value is 0 leave it alone
            for language in self.smooth_probability_log.keys():
                self.smooth_probability_log[language] = math.log((smoothing_value/float(self.ngram_portion[language] + smoothing_value*(math.pow(self.vocab_list_size, n_gram_size)))), 10)
        
        # now that we have models with the probabilities, we want to look at the test set to see if we can determine the
        # language of each tweet.
        # read from the test files and try determining the language
        f = open(test_name, "r", encoding="utf-8")
        fileInput = f.readline()

        if(vocabulary==3):
            vocab_name = "BYOM"
        else:
            vocab_name = str(vocabulary)
        trace_filename = str("trace_" + vocab_name + "_" + str(n_gram_size) + "_" + str(smoothing_value)+".txt")
        solution_trace_file = open(trace_filename, "w", encoding="utf-8")  # this is the trace file that the program will write to
        overall_eval_filename = str("eval_" + vocab_name + "_" + str(n_gram_size) + "_" + str(smoothing_value)+".txt")
        overall_eval_file = open(overall_eval_filename, "w", encoding="utf-8")  # this is the eval file that the program will write to
        total_lines = 0
        total_correct = 0

        ca_test_language_count = 0
        eu_test_language_count = 0
        es_test_language_count = 0
        en_test_language_count = 0
        gl_test_language_count = 0
        pt_test_language_count = 0

        # this is the loop that reads the file input line by line
        while fileInput:
            # print(fileInput)
            tweet_id = fileInput.split("\t")[0]
            tweet_author = fileInput.split("\t")[1]
            tweet_language = fileInput.split("\t")[2]
            tweet = fileInput.split("\t")[3]

            if tweet_language == "eu":
                eu_test_language_count += 1
            elif tweet_language == "ca":
                ca_test_language_count += 1
            elif tweet_language == "es":
                es_test_language_count += 1
            elif tweet_language == "en":
                en_test_language_count += 1
            elif tweet_language == "gl":
                gl_test_language_count += 1
            elif tweet_language == "pt":
                pt_test_language_count += 1



            if vocabulary == 3:
                tweet = self.clean_string(tweet)

            language = self.determine_language(tweet)

            self.actual_language_count[tweet_language] += 1

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

        print("Test file results: ")
        print("ca: " + str(ca_test_language_count))
        print("eu: " + str(eu_test_language_count))
        print("es: " + str(es_test_language_count))
        print("en: " + str(en_test_language_count))
        print("gl: " + str(gl_test_language_count))
        print("pt: " + str(pt_test_language_count))

        # calculate the accuracy
        accuracy = float(total_correct / total_lines)
        overall_eval_file.write(str('{:.4f}'.format(accuracy)) + "\n")

        # calculate the precision
        # precision = TP / (TP + FP) = (# of correct guesses by L) / (# of correct + # of wrong)
        # TP: Your system designated a tweet a certain language. The tweet is in fact that language.
        # FP: Your system designated a tweet a certain langauge. The tweet is not actually that language. The system is wrong
        for lang in self.models:
            if self.tp_count[lang] == 0:
                precision = 0
            else:
                precision = self.tp_count[lang] / (self.tp_count[lang] + self.fp_count[lang])
            self.precision_values[lang] = precision
            overall_eval_file.write(str('{:.4f}'.format(precision)) + "  ")
        overall_eval_file.write("\n")

        # calculate the recall
        # recall = TP / (TP + FN) = (# correct guesses by L) / (# of times the actual answer was L)
        for lang in self.models:
            if self.tp_count[lang] == 0:
                recall = 0
            else:
                recall = self.tp_count[lang] / self.actual_language_count[lang]
            self.recall_values[lang] = recall
            overall_eval_file.write(str('{:.4f}'.format(recall)) + "  ")
        overall_eval_file.write("\n")

        # calcualte the F1 measure
        # F = (B^2 + 1)PR/(B^2P+R) -> F1 means B = 1
        for lang in self.models:
            if self.precision_values[lang] == 0 or self.recall_values[lang] == 0:
                f1 = 0
            else:
                f1 = float((2 * self.precision_values[lang] * self.recall_values[lang]) / (self.precision_values[lang] + self.recall_values[lang]))
            self.f1_values[lang] = f1
            overall_eval_file.write(str('{:.4f}'.format(f1)) + "  ")
        overall_eval_file.write("\n")

        # calcualte the macro-F1 and weighed-average-F1
        # macro-F1 is just the average of all F1 values
        macro_f1_sum = 0.0
        macro_f1_count = 0
        for lang in self.models:
            macro_f1_count += 1
            macro_f1_sum += self.f1_values[lang]
        macro_f1_value = float(macro_f1_sum / macro_f1_count)

        # weighted average is using the count of each of the languages in the test file
        macro_weighted_count = 0
        macro_weighted_sum = 0.0
        for lang in self.models:
            macro_weighted_count += self.actual_language_count[lang]
            macro_weighted_sum += self.actual_language_count[lang] * self.f1_values[lang]
        macro_weighted_value = float(macro_weighted_sum / macro_weighted_count)

        overall_eval_file.write(str('{:.4f}'.format(macro_f1_value)) + "  " + str('{:.4f}'.format(macro_weighted_value)) + "\n")

        solution_trace_file.close()
        overall_eval_file.close()
        f.close()
    
    def clean_string(self, string):
        # remove any Twitter usernames and urls from string
        stringClean = re.sub(r'@.+?\s', '', string)      # regular case of username followed by whitespace
        stringClean = re.sub(r'@.+?$', '', stringClean)      # edge case where last 'word' is a username
        stringClean = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', stringClean)
        # above line for cleaning urls by kerim of stackoverflow https://stackoverflow.com/questions/14081050/remove-all-forms-of-urls-from-a-given-string-in-python
        return stringClean

    def determine_language(self, tweet):
        best_probability = -100.0  # 1.0 is the highest possible probability so we will update this value with the calculated results
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

        for gram in tweet_gram:
            is_gram_in_vocab = True
            for j in range(self.n_gram_size):  # first determine if the gram is in the vocabulary. If not, we wont consider it
                if (self.vocabulary == 2 and not gram[j].isalpha()):     
                    is_gram_in_vocab = False    # for vocabulary 2, any isalpha is in vocab
                elif (self.vocabulary != 2 and gram[j] not in self.vocab_list):
                    is_gram_in_vocab = False
            if is_gram_in_vocab:  # if the gram is actually in the vocabulary
                if gram not in self.models[model]:  # if the gram is not in the model
                    probability = probability + self.smooth_probability_log[model]
                else:  # if the gram is in the model
                    probability = probability + math.log(self.models[model][gram], 10)  # using the addition of log base 10 for the probability

        # note to self.
        # log(P(A,B,C)) = log(P(A)) + log(P(B)) + log(P(C))
        # P(A,B,C) = e ^ log(P(A,B,C))
        self.trace_probability = probability
        return math.exp(probability)

if __name__ == "__main__":
    nlp = NaturalLanguageProcessing()

    #nlp.produce_output(0, 1, 0.5, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.6831
    #nlp.produce_output(1, 1, 0.5, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.6977
    #nlp.produce_output(2, 1, 0.5, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.5676
    #nlp.produce_output(0, 2, 0.5, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.8420
    #nlp.produce_output(1, 2, 0.5, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.8441
    #nlp.produce_output(2, 2, 0.5, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.5676
    #nlp.produce_output(0, 3, 0.5, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.9029
    #nlp.produce_output(1, 3, 0.5, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.7704
    #nlp.produce_output(2, 3, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    #nlp.produce_output(3, 3, 0.5, "training-tweets.txt", "test-tweets-given.txt")

    nlp.produce_output(0, 2, 0.0, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(0, 2, 0.3, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(0, 2, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(0, 2, 0.9, "training-tweets.txt", "test-tweets-given.txt")

    nlp.produce_output(0, 3, 0.0, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(0, 3, 0.3, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(0, 3, 0.5, "training-tweets.txt", "test-tweets-given.txt")
    nlp.produce_output(0, 3, 0.9, "training-tweets.txt", "test-tweets-given.txt")



    #nlp.produce_output(3, 3, 0.0, "training-tweets.txt", "test8.txt") # accuracy = 85.02
    #nlp.produce_output(3, 3, 0.3, "training-tweets.txt", "test8.txt") # accuracy = 85.02
    #nlp.produce_output(3, 3, 0.5, "training-tweets.txt", "test8.txt") # accuracy = 85.02
    #nlp.produce_output(3, 3, 0.9, "training-tweets.txt", "test8.txt") # accuracy = 85.02

    #nlp.produce_output(1, 3, 0.5, "training-tweets.txt", "test8.txt")
    #nlp.produce_output(2, 3, 0.5, "training-tweets.txt", "test8.txt")
    #nlp.produce_output(3, 3, 0.5, "training-tweets.txt", "test8.txt")

    #nlp.produce_output(0, 1, 0.2, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.6831
    #nlp.produce_output(1, 1, 0.2, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.6970
    #nlp.produce_output(2, 1, 0.2, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.5676
    #nlp.produce_output(0, 2, 0.2, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.8420
    #nlp.produce_output(1, 2, 0.2, "training-tweets.txt", "test-tweets-given.txt")  # accuracy = 0.8451
    #nlp.produce_output(2, 2, 0.2, "training-tweets.txt", "test-tweets-given.txt")  # accuracy =
    #nlp.produce_output(0, 3, 0.2, "training-tweets.txt", "test-tweets-given.txt")  # accuracy =
    #nlp.produce_output(1, 3, 0.2, "training-tweets.txt", "test-tweets-given.txt")  # accuracy =
    #nlp.produce_output(2, 3, 0.2, "training-tweets.txt", "test-tweets-given.txt")  # accuracy =

    #nlp.produce_output(0, 1, 0, "training-tweets.txt", "test-tweets-given.txt")  # These 5 are the required test sets
    #nlp.produce_output(1, 2, 0.5, "training-tweets.txt", "test-tweets-given.txt")  
    #nlp.produce_output(1, 3, 1, "training-tweets.txt", "test-tweets-given.txt")  
    #nlp.produce_output(2, 2, 0.3, "training-tweets.txt", "test-tweets-given.txt")  
    #nlp.produce_output(3, 3, 0.01, "training-tweets.txt", "test-tweets-given.txt") # This is the BYOM as submitted, with optimal smoothing value for this test data

    #nlp.produce_output(3, 3, 0.3, "training-tweets.txt", "test-tweets-given.txt")  # This is the BYOM with generally most optimal parameters
    #nlp.produce_output(3, 3, 0.3, "training-tweets.txt", "test8.txt")  # This is the BYOM with generally most optimal parameters


    #nlp.produce_output(0, 2, 0.01, "training-tweets.txt", "test8.txt")  # test 1
    #nlp.produce_output(2, 2, 0.5, "training-tweets.txt", "test8.txt")   # test 2
    #nlp.produce_output(3, 3, 0.3, "training-tweets.txt", "test8.txt")   # BYOM test

    #1 - V = 0, n = 2, d = 0.01
    #2 - V = 2, n = 2, d = 0.5
    #3 - BYOM


    print("WE ARE DONE WITH THE TESTS")

