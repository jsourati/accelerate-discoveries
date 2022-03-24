import os
import sys
import pdb
import json
import regex
import logging
import numpy as np
from tqdm import tqdm
from scipy import sparse
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.phrases import Phrases, Phraser

import utils

DEFAULT_PARS_PATH = os.path.join(os.path.dirname(__file__), "default_params.json")
with open(DEFAULT_PARS_PATH, 'r') as f:
    DEFAULT_PARS = json.load(f)
exclude_terms = [":", "=", ".", ",", "(", ")", "<", ">", "\"", "“", "”", "≥", "≤", "<nUm>"]


class dww2v(object):
    '''Class of Word2Vec embedding function that is mostly suitable for learning
    embedding from random walk sequences (within a deepwalk framework)
    '''

    def __init__(self, path_to_data, **kwargs):


        self.path_to_data = path_to_data
        self.pars = {}
        for key, def_val in DEFAULT_PARS.items():
            self.pars[key] = kwargs.get(key, def_val)

        # setting up the logger
        logger_disable = kwargs.get('logger_disable', False)
        self.logfile_path =   kwargs.get('logfile_path', None)
        self.logger = utils.set_up_logger(__name__, self.logfile_path, logger_disable)

        
    def load_model(self, path):
        self.model = Word2Vec.load(path)

        
    def build_model(self, phrasing=True):

        self.logger.info('Parsing lines (sentences) in: {}: '.format(self.path_to_data))
        self.logger.info('Parameters for parsing phrases are as follows:')
        for key in ['depth', 'phrase_min_count', 'phrase_threshold']:
            self.logger.info('\t{}: {}'.format(key, self.pars[key]))

        
        self.sentences = LineSentence(self.path_to_data)

        if phrasing:
            self.sentences, self.phrases = extract_phrases(self.sentences,
                                                           self.pars['depth'],
                                                           self.pars['phrase_min_count'],
                                                           self.pars['phrase_threshold'])

        # build the embedding model        
        self.model = Word2Vec(self.sentences,
                              size=self.pars['size'],
                              window=self.pars['window'],
                              min_count=self.pars['min_count'],
                              sg=self.pars['sg'],
                              hs=self.pars['hs'],
                              workers=self.pars['workers'],
                              alpha=self.pars['start_alpha'],
                              sample=self.pars['subsample'],
                              negative=self.pars['negative'],
                              compute_loss=True,
                              sorted_vocab=True,
                              batch_words=self.pars['batch'],
                              iter=0   # this will be "epochs" in newer versions
                              )

        

    def train(self, **kwargs):

        self.model_save_path = kwargs.get('model_save_path', None)
        brkpnt = kwargs.get('brkpnt', 1)
        
        callbacks = [MyCallBack(brkpnt, self.model_save_path, self.logger)]

        self.logger.info('Training the model using the following parameters:')
        for key, val in self.pars.items():
            if key in ['depth', 'phrase_count', 'phrase_threshold']: continue
            self.logger.info('\t{}: {}'.format(key, val))
        self.logger.info('The model will be saved in {}'.format(self.model_save_path))

        self.model.train(self.sentences,
                         total_examples=self.model.corpus_count,
                         start_alpha=self.pars['start_alpha'],
                         end_alpha=self.pars['end_alpha'],
                         epochs=self.pars['epochs'],
                         compute_loss=True,
                         callbacks=callbacks)

        
    def most_similar_props(self, token, prop=None):
        '''Finding the most similar properties to a given token
        '''
        pass
        


def extract_phrases(sent, depth, min_count, threshold, level=0):
    '''Extracting phrases from the corpus (inspired by `mat2vec.training.phrase2vec.wordgrams`)
    '''
    
    if depth == 0:
        return sent, None
    else:
        phrases = Phrases(sent,
                          min_count=min_count,
                          threshold=threshold)

        phrases = Phraser(phrases)
        phrases.phrasegrams = exclude_words(phrases.phrasegrams, exclude_terms)
        level += 1
        if level < depth:
            return extract_phrases(phrases[sent], depth, min_count, threshold, level)
        else:
            return phrases[sent], phrases


def exclude_words(phrasegrams, words):
    """Given a list of words, excludes those from the keys of the phrase dictionary."""
    new_phrasergrams = {}
    words_re_list = []
    for word in words:
        we = regex.escape(word)
        words_re_list.append("^" + we + "$|^" + we + "_|_" + we + "$|_" + we + "_")
    word_reg = regex.compile(r""+"|".join(words_re_list))
    for gram in tqdm(phrasegrams):
        valid = True
        for sub_gram in gram:
            if word_reg.search(sub_gram.decode("unicode_escape", "ignore")) is not None:
                valid = False
                break
            if not valid:
                continue
        if valid:
            new_phrasergrams[gram] = phrasegrams[gram]
    return new_phrasergrams



class MyCallBack(CallbackAny2Vec):

    """Callback to save model after every epoch."""
    def __init__(self, brkpnt=1, model_save_path=None, logger=None):
        self.epoch = 0
        self.losses = []
        #self.man_acc = []
        self.brkpnt = brkpnt
        self.logger = logger
        self.model_save_path = model_save_path

    def on_epoch_end(self, model):
        self.epoch += 1
        if not(self.epoch%self.brkpnt):
            if self.epoch==1:
                self.losses += [model.get_latest_training_loss()]
            else:
                self.losses += [model.get_latest_training_loss() - self.last_loss]

            self.last_loss = model.get_latest_training_loss()
            # manually added evaluator
            #self.man_acc += [self.man_eval(model)]

            if self.model_save_path is not None:
                if self.epoch==1:
                    model.save(self.model_save_path)
                else:
                    if self.losses[-1] < np.min(self.losses[:-1]):
                        model.save(self.model_save_path)

            if self.logger is not None:
                self.logger.info('{} Epoch(s) done. Loss: {}, LR: {}'.format(self.epoch,
                                                                             self.losses[-1],
                                                                             model.min_alpha_yet_reached))
    

