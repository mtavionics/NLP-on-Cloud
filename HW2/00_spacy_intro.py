#!/usr/bin/env python
# coding: utf-8

# # [spaCy](http://spacy.io/docs/#examples) introduction

# ## Load spaCy resources

# # Import spacy and English models
# import spacy
# 
# nlp = spacy.load('en')

# In[ ]:


Loading spaCy can take a while, in the meantime here are a few definitions to help you on your NLP journey.

#### What are Stop Words?

Stop words are the common words in a vocabulary which are of little value when considering word frequencies in text. This is because they don't provide much useful information about what the sentence is telling the reader.

Example: _"the","and","a","are","is"_

#### What is a Corpus?

A corpus (plural: corpora) is a large collection of text or documents and can provide useful training data for NLP models. A corpus might be built from transcribed speech or a collection of manuscripts. Each item in a corpus is not necessarily unique and frequency counts of words can assist in uncovering the structure in a corpus.

Examples:

1. Every word written in the complete works of Shakespeare
2. Every word spoken on BBC Radio channels for the past 30 years 


# ## Process text

# In[ ]:


# Process sentences 'Hello, world. Natural Language Processing in 10 lines of code.' using spaCy
doc = nlp(u'Hello, world. Natural Language Processing in 10 lines of code.')


# ## Get tokens and sentences
# 
# #### What is a Token?
# A token is a single chopped up element of the sentence, which could be a word or a group of words to analyse. The task of chopping the sentence up is called "tokenisation".
# 
# Example: The following sentence can be tokenised by splitting up the sentence into individual words.
# 
# 	"Cytora is going to PyCon!"
# 	["Cytora","is","going","to","PyCon!"]

# In[ ]:


# Get first token of the processed document
token = doc[0]
print(token)

# Print sentences (one sentence per line)
for sent in doc.sents:
    print(sent)


# ## Part of speech tags
# 
# #### What is a Speech Tag?
# A speech tag is a context sensitive description of what a word means in the context of the whole sentence.
# More information about the kinds of speech tags which are used in NLP can be [found here](http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/).
# 
# Examples:
# 
# 1. CARDINAL, Cardinal Number - 1,2,3
# 2. PROPN, Proper Noun, Singular - "Matic", "Andraz", "Cardiff"
# 3. INTJ, Interjection - "Uhhhhhhhhhhh"

# In[ ]:


# For each token, print corresponding part of speech tag
for token in doc:
    print('{} - {}'.format(token, token.pos_))


# ## Visual part of speech tagging ([displaCy](https://displacy.spacy.io))

# ## Syntactic dependencies
# 
# #### What are syntactic dependencies?
# 
# We have the speech tags and we have all of the tokens in a sentence, but how do we relate the two to uncover the syntax in a sentence? Syntactic dependencies describe how each type of word relates to each other in a sentence, this is important in NLP in order to extract structure and understand grammar in plain text.
# 
# Example:
# 
# <img src="https://github.com/explosion/spacy-notebooks/blob/master/notebooks/conference_notebooks/pycon_nlp/images/syntax-dependencies-oliver.png?raw=1" align="left" width=500>

# In[ ]:


# Write a function that walks up the syntactic tree of the given token and collects all tokens to the root token (including root token).

def tokens_to_root(token):
    """
    Walk up the syntactic tree, collecting tokens to the root of the given `token`.
    :param token: Spacy token
    :return: list of Spacy tokens
    """
    tokens_to_r = []
    while token.head is not token:
        tokens_to_r.append(token)
        token = token.head
        tokens_to_r.append(token)

    return tokens_to_r

# For every token in document, print it's tokens to the root
for token in doc:
    print('{} --> {}'.format(token, tokens_to_root(token)))

# Print dependency labels of the tokens
for token in doc:
    print('-> '.join(['{}-{}'.format(dependent_token, dependent_token.dep_) for dependent_token in tokens_to_root(token)]))


# ## Named entities
# 
# #### Named Entities
# 
# A named entity is any real world object such as a person, location, organisation or product with a proper name. 
# 
# Example:
# 
# 	1. Barack Obama
# 	2. Edinburgh
# 	3. Ferrari Enzo

# In[ ]:


# Print all named entities with named entity types

doc_2 = nlp(u"I went to Paris where I met my old friend Jack from uni.")
for ent in doc_2.ents:
    print('{} - {}'.format(ent, ent.label_))


# ## Noun chunks
# 
# #### What is a Noun Chunk?
# Noun chunks are the phrases based upon nouns recovered from tokenized text using the speech tags.
# 
# Example:
# 
# The sentence "The boy saw the yellow dog" has 2 noun objects, the boy and the dog. 
# Therefore the noun chunks will be
# 
# 	1. "The boy"
# 	2. "the yellow dog"

# In[ ]:


# Print noun chunks for doc_2
print([chunk for chunk in doc_2.noun_chunks])


# ## Unigram probabilities

# In[ ]:


# For every token in doc_2, print log-probability of the word, estimated from counts from a large corpus 
for token in doc_2:
    print(token, ',', token.prob)


# ## Word embedding / Similarity
# 
# #### What are Word embeddings?
# 
# A word embedding is a representation of a word, and by extension a whole language corpus, in a vector or other form of numerical mapping. This allows words to be treated numerically with word similarity represented as spatial difference in the dimensions of the word embedding mapping.
# 
# Example:
# 	
# With word embeddings we can understand that vector operations describe word similarity. This means that we can see vector proofs of statements such as:
# 
# 	king-queen==man-woman

# In[ ]:


# For a given document, calculate similarity between 'apples' and 'oranges' and 'boots' and 'hippos'
doc = nlp(u"Apples and oranges are similar. Boots and hippos aren't.")
apples = doc[0]
oranges = doc[2]
boots = doc[6]
hippos = doc[8]
print(apples.similarity(oranges))
print(boots.similarity(hippos))

print()
# Print similarity between sentence and word 'fruit'
apples_sent, boots_sent = doc.sents
fruit = doc.vocab[u'fruit']
print(apples_sent.similarity(fruit))
print(boots_sent.similarity(fruit))


# 
