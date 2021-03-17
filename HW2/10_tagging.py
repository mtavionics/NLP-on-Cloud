# -*- coding: utf-8 -*-
"""
Created on Sun Nov 6 21:07:30 2020

@ Mikhail Terentev

10.	Explain Parts-of-speech Tagging 

"""
import nltk
from nltk.tokenize import word_tokenize

text = word_tokenize("How to check word similarity using the spacy package? Review and run following code and explain.")

# Split text into words
tokens = nltk.pos_tag(text)

print (tokens)

