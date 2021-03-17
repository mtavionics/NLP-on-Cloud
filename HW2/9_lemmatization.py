# -*- coding: utf-8 -*-
"""
Created on Sun Nov 6 21:07:30 2020

@ Mikhail Terentev

9.	Explain Lemmatization 

"""

from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("studies"))

print(lemmatizer.lemmatize("feet"))
