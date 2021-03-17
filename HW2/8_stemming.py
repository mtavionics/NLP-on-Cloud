# -*- coding: utf-8 -*-
"""
Created on Sun Nov 6 21:07:30 2020

@ Mikhail Terentev

8. Explain Stemming with the help of an example.

"""

from nltk.stem import PorterStemmer

porter = PorterStemmer()

print(porter.stem('CONNECTING'))
print(porter.stem('plays'))
print(porter.stem('playing'))

