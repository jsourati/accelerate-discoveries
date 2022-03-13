import os
import re
import sys
import pdb
import json
import random
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
