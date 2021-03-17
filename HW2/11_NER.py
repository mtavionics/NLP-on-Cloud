# -*- coding: utf-8 -*-
"""
Created on Sun Nov 6 21:07:30 2020

@ Mikhail Terentev

11.	Explain NER 

"""
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Netflix leads on downloads, but YouTube Kids grabs more hours in Washington"
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)




