import os
import re
import sys
import pdb
import json
import random
import logging
import numpy as np


def extract_props(fpath):
    
    props = json.load(open(fpath,'r'))
    props = [['{}/{}'.format(x.replace(' ','__'),y.replace(' ','__')) for y in props[x]] for x in props]

    return props


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

    newseq = [x for x in newseq if len(np.unique(x.split(' ')))>1]

    return newseq


def set_up_logger(log_name, logfile_path, logger_disable, file_mode='w'):
    """Setting up handler of the "root" logger as the single main logger

    If `logger_disable` is `True`, no logging of any kind will be done disregarding other inputs.
    Otherwise, if `logfile_path` is `None`, the logging will be done only through the root
    logger in a streaming format. Finally, if `logfile_path` is set to a path string, the
    logs will be stored to a file rather than streaming.
    """
    
    logger = logging.getLogger(log_name)
    if logger_disable:
        logger.handlers = []
        logging.root.handlers = []
    elif logfile_path is None:
        logging.root.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s : %(levelname)s : %(message)s"))
        logging.root.handlers = [handler]
        logger.handlers = []
    else:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename=logfile_path,
                                      encoding='utf-8',
                                      mode=file_mode)
        handler.setFormatter(logging.Formatter("%(asctime)s : %(levelname)s : %(message)s"))
        logger.handlers = []
        logger.addHandler(handler)

    return logger
