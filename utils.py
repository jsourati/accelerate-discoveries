import os
import re
import sys
import pdb
import json
import random
import logging
import numpy as np



def remove_authors_from_RW(seq):
    '''Removing author nodes from a list of RW sequences
    '''

    patt1 = "a_(\d)+( )"   # for all authors but the last one in the seq (if any)
    patt2 = "( )a_(\d)+"   # for the last author
    
    newseq = []
    for sq in seq:
        nsq = re.sub(patt1,"",sq)
        nsq = re.sub(patt2,"",nsq)
        newseq += [nsq]

    return newseq


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



def set_up_logger(log_name, logfile_path, logger_disable, file_mode='w'):
    """Setting up handler of the "root" logger as the single main logger
    """
    
    logger = logging.getLogger(log_name)
    if logger_disable:
        handler = logging.NullHandler()
    elif logfile_path is None:
        handler = logging.StreamHandler()
    else:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename=logfile_path,
                                      encoding='utf-8',
                                      mode=file_mode)
    handler.setFormatter(logging.Formatter("%(asctime)s : %(levelname)s : %(message)s"))
    logger.handlers = []
    logger.addHandler(handler)

    return logger
