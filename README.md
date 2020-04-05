# COMP-472-P2
This repository contains all the code for AI project 2 winter 2020.

Group members 
Vincent Cerri - 40034135


How to use:
This program is using python v3.7. 

This program reads tweets from a test file which has the following format:

ID \t username \t language \t tweet 

To run the program, there must be a training file for the program to build a model, and a test file for the program to test the created models. 
Use the produce_output(V, size, smoothing_value, training_name, testing_name) method in the main method with the following variables:

V = integer value representing the vocabulary to be used.

  0 -> Fold the corpus to lowercase and use only the 26 letters of the alphabet [a-z]

  1 -> Distinguish up and low cases and use only the 26 letters of the alphabet [a-z, A-Z]

  2 -> Distinguish up and low cases and use all characters accepted by the built-in isalpha() method

  3 -> Our custom BYOM 

size = N-gram size indicates into how many characters we should be splitting up the strings

  1 -> unigrams

  2 -> bigrams

  3 -> trigrams

smoothing_value = a real in the interval [0...1]

training_name = the name of the file containing the training set. 

testing_name = the name of the file containing the testing set. 

This program will produce 2 output files. Trace and Eval. 
