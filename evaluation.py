import os
import re
import sys
import pdb
import json
import random
import logging
import numpy as np


def cosine_sims(model, tokens_A, token_B, type='embed_out'):
    """
    Calculating cosine similarities between a set of tokens A and a single token B based
    on a trained model. When `type` input is set to `embed_out` the cosine similarity is 
    computed between output weights for tokens A and the hidden weights for token B.
    

    Note the following vectors in a gensim's word2vec model:

    model.wv.vectors, model.wv.syn0 and model.wv[word]:
        all these three give word embedding vectors, the first two use
        word indices and the last use the word's in string format to
        return the embedding vector
        ---SANITY CHECK---
        these three vectors are the same:
        model.wv.vectors[i,:], model.wv.syn0[i,:], model.wv[model.wv.index2word[i]]

    model.wv.trainables.syn1neg:
        output embedding used in negative sampling (take index, not string value)

    model.wv.trainables.syn1:
        output embedding used in heirarchical softmax
    """

    if token_B not in model.wv.vocab:
        raise NameError("{} is not in the model's vocabulary.".format(token_B))

    # embedding vector of B --> hidden weights
    zw_y = model.wv[token_B]
    zw_y = zw_y / np.sqrt(np.sum(zw_y**2))
    
    sims = np.ones(len(tokens_A))
    for i,tok in enumerate(tokens_A):
        # if a token is not in the vocabulary --> sim.=NaN
        if tok not in model.wv.vocab:
            sims[i] = np.nan
            continue

        idx = model.wv.vocab[tok].index
        if type=='embed_out':
            zo_x = model.trainables.syn1neg[idx,:]
        elif type=='embed_embed':
            zo_x = model.wv[tok]
            
        zo_x = zo_x / np.sqrt(np.sum(zo_x**2))

        sims[i] = np.dot(zw_y, zo_x)

    return sims


def props_stats(model, props):

    res_dict = {}
    for prop in props:
        cnt = {}
        for val in props[prop]:
            tok = '{}/{}'.format(prop.replace(' ','__'),val.replace(' ','__'))
            if tok in model.wv.vocab:
                cnt[val] = model.wv.vocab[tok].count
            else:
                cnt[val] = 0

        res_dict[prop] = cnt

    return res_dict


def infer_props(model, mat, props):
    """Inferring property for a certain material. 

    `The input properties should be a dictionary with keys as the property names
    and values as all possible attributes to that property.
    """

    infers = {}
    scores = {}
    for prop,vals in props.items():
        toks = ['{}/{}'.format(prop,val) for val in vals]
        sims = cosine_sims(model, toks, mat)
        scores[prop] = {vals[i]:sims[i] for i in np.argsort(-sims)}
        
        if not(np.all(np.isnan(sims))):
            infers[prop] = vals[np.argsort(-sims)[0]]
        else:
            infers[prop] = np.nan

    return scores,infers

def infer_elements(model, token, elements):
    """Inferring elements in regards with a given token through a trained model
    """

    model_elements = np.array([x for x in elements if x in model.wv.vocab])
    sims = cosine_sims(model, model_elements, token)
    out_elements = model_elements[np.argsort(-sims)]
    out_sims = sims[np.argsort(-sims)]

    return out_elements, out_sims
    
