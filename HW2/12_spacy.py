# -*- coding: utf-8 -*-
"""
Created on Sun Nov 6 21:07:30 2020

@ Mikhail Terentev

12.	Review and run following code and explain

"""
import spacy

nlp = spacy.load('en_core_web_md') 
print("Enter the words") 
input_words = input()
tokens = nlp(input_words) 
for i in tokens:
    print(i.text, i.has_vector, i.vector_norm, i.is_oov) 
    token_1, token_2 = tokens[0], tokens[1]
    print("Similarity between words:", token_1.similarity(token_2))



