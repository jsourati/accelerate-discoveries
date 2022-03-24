import os
import sys
import pdb
import json
import random
import logging
import pymysql
import numpy as np
import networkx as nx
from scipy import sparse
from collections import deque
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count


class hypergraph(object):

    def __init__(self, R, mats, props, authors=None):
        '''Make sure to feed the weight matrix in a format such that 
        the first batch of columns (nA) corresponds to authors, the second
        batch (nM) corresponds to the pool of materials and the third batch (nP)
        corresponds to properties
        '''
        
        self.R = R
        self.mats = mats
        self.props = props
        self.nM = len(mats)
        self.nP = len(props)
        self.nA = R.shape[1] - self.nM - self.nP
        if authors is not None:
            assert len(authors)==self.nA, "Number of author names should match " + \
                "the number of clumns in the node matrix."

            
    def get_csr_mat(self):
        self.Rcsr = self.R.tocsr()
            
            
    def find_neighbors(self, idx, return_names=False):
        """Returning neighbors of a node indexed by `idx`

        NOTE: input `idx` can be an array of indices
        """

        # indices of the hyperedges (there might be repeated hyperedges
        # here, if idx is an array, but we don't care since the final
        # result is distinct values of the column list)
        he_inds = self.R[:,idx].nonzero()[0]

        nbrs = np.unique(self.R[he_inds,:].nonzero()[1])
        if return_names:
            return [self.idx_to_name(x) for x in nbrs]
        else:
            return nbrs

    def paper_to_nodes(self,idx, return_names=False):
        '''Returining nodes that are involved in a given paper identified
        by its row index
        '''

        nodes_inds = self.R[idx,:].nonzero()[1]

        if return_names:
            return [self.idx_to_name(x) for x in nodes_inds]
        else:
            return nodes_inds

        
    def node_to_papers(self,idx):
        '''Returning all papers that include a given node
        '''
        return self.R[:,idx].nonzero()[0]

    def mat_to_papers(self,name):
        '''Returning all papers that include a given material identified
        by its name string
        '''
        assert name in self.mats, "{} could not be found.".format(name)
        idx = self.nA + np.where(self.mats==name)[0][0]
        return self.R[:,idx].nonzero()[0]
    
    def compute_transprob(self):
        """Computing the transition probability matrix given the
        binary (0-1) vertex weight matrix (dim.; |E|x|V|)
        """

        row_collapse = np.array(np.sum(self.R,axis=0))[0,:]
        iDV = np.zeros(len(row_collapse), dtype=float)
        iDV[row_collapse>0] = 1./row_collapse[row_collapse>0]
        iDV = sparse.diags(iDV, format='csr')

        col_collapse = np.array(np.sum(self.R,axis=1))[:,0]
        iDE = np.zeros(len(col_collapse), dtype=float)
        iDE[col_collapse>0] = 1./col_collapse[col_collapse>0]
        iDE = sparse.diags(iDE, format='csr')

        #         edge sel.      node sel.
        #           prob.          prob.
        #      --------------   ------------
        return iDV * self.R.T * iDE * self.R


    def idx_to_name(self, idx):
        ''' Translating a node ID to a string indicating type of the node
        based on the order of the columns in our vertex weight matrix
        (e.g., node 150 --> the 150-th author, or material "CO2")
        '''
        

        if idx < self.nA:
            return 'a_{}'.format(idx)
        elif self.nA <= idx < (self.nA+self.nM):
            return self.mats[idx-self.nA]
        elif (self.nA+self.nM) <= idx:
            return self.props[idx-self.nA-self.nM]
        else:
            raise ValueError("Index {} is not in any expected interval.".format(idx))


    def alpha_modify_dist(self, alpha, hotvec):
        '''Modifying a uniform (sampling) distribution over nodes based on
        our alpha-adjusted non-uniform weights

        alpha = P(mats) / P(authors or props)

        The input `hot_vec` is a one-hot vector that specifies the
        nodes that could be sampled in the current sampling step.

        Here it is assumed that `nA,nM,nP` are all non-zero (integer) values.
        '''

        if np.all(hotvec==0):
            return hotvec

        if alpha==np.inf:
            hotvec[:self.nA] = 0
        
        sum_AP = np.sum(hotvec[:self.nA]) + np.sum(hotvec[self.nA+self.nM:])
        sum_M = np.sum(hotvec[self.nA:self.nA+self.nM])

        if sum_AP>0:
            hotvec[:self.nA] = hotvec[:self.nA] / sum_AP
            hotvec[self.nA+self.nM:] = hotvec[self.nA+self.nM:] / sum_AP
        if sum_M>0:
            hotvec[self.nA:self.nA+self.nM] = alpha*hotvec[self.nA:self.nA+self.nM] / sum_M
        

        return hotvec/np.sum(hotvec)

    
    def random_walk(self,length,size,**kwargs):
        """Generating a sequence of random walks over the hypergraph

        Input argument block_types specifies type of the "column blocks" in the vertex
        matrix, with format ((B1,n1), (B2,n2),...), where Bi and ni are the i-th block and
        its size. It is assumed that these blocks are grouped in the same order as in
        this variable(they are not shuffled).

        Input `alpha` is either a scalar that determines the ratio of the probability of 
        choosing a material to the probability of author selection (if two types
        of nodes are present), or an array-like that determines mixture coefficients
        corresponding to various groups of nodes (if multiples types of nodes are present)

        The argument `block_types` determines groups of columns that exist in the given
        vertex matrix R. It should be given as a dictionary with a format like the following:
        {'author': nA, 'entity': nE}, where nA and nE are the number of author nodes and
        entity nodes, respectively.
        """
        
        alpha = kwargs.get('alpha', None)
        start_inds = kwargs.get('start_inds', None)
        node2vec_q = kwargs.get('node2vec_q', None)
        nseq_file_path = kwargs.get('nseq_file_path',None)
        eseq_file_path = kwargs.get('eseq_file_path',None)
        rand_seed = kwargs.get('rand_seed',None)

        
        # setting the initial node index    
        if start_inds is None:
            n = self.R.shape[1]
            init_idx = np.random.randint(0,n,length) if rand_seed is None else \
                np.random.RandomState(rand_seed).randint(0,n,length)
        elif isinstance(start_inds, (list,np.ndarray)):
            # randomly choose one of them
            rand_idx = np.random.randint(0,len(start_inds),length) if rand_seed is None \
                else np.random.RandomState(rand_seed).randint(0,len(start_inds),length)
            init_idx = [start_inds[x] for x in rand_idx]
        elif np.isscalar(start_inds):
            if start_inds-int(start_inds) != 0:
                raise ValueError("The starting index in a random walk should be " +\
                                 "a positive integer (not a float like {}).".format(start_inds))
            init_idx = np.ones(length,dtype=int) * int(start_inds)
                    

        # setting up the sampling distribution
        if alpha is None:
            # uniform sampling
            f = None
        elif np.isscalar(alpha):
            # alpha-modified sampling
            f = lambda data: self.alpha_modify_dist(alpha, data)

            
        if rand_seed is None:
            rand_seeds = [None]*size
        else:
            rand_seeds = rand_seed + np.arange(size)

        Rcsr = self.Rcsr if hasattr(self,'Rcsr') else None
            
        sents = []
        eseqs_list = []
        nlines=0
        for i in range(size):
            seq, eseq = random_walk_for_hypergraph(self.R,
                                                   init_idx[i],
                                                   length,
                                                   lazy=False,
                                                   node_weight_func=f,
                                                   node2vec_q=node2vec_q,
                                                   rand_seed=rand_seeds[i],
                                                   Rcsr=Rcsr)
            
            eseqs_list += [' '.join([str(x) for x in eseq])]

            # parsing the hyper nodes
            toks = [self.idx_to_name(s) for s in seq]
            sent = ' '.join(toks)

            sents += [sent]

            if not(i%500) and i>0:
                if file_path is not None:
                    with open(file_path, 'a') as tfile:
                        tfile.write('\n'.join(sents[i-500:i])+'\n')
                        nlines = i
                if eseq_file_path:
                    with open(eseq_file_path, 'a') as tfile:
                        tfile.write('\n'.join(eseqs_list[i-500:i])+'\n')
                        nlines = i
                if logger is not None:
                    logger.info('{} randm walks are saved'.format(i))

        if nseq_file_path is not None:
            with open(nseq_file_path, 'a') as f:
                f.write('\n'.join(sents[nlines:])+'\n')
        if eseq_file_path is not None:
            with open(eseq_file_path, 'a') as f:
                f.write('\n'.join(eseqs_list[nlines:])+'\n')


        return sents, eseqs_list

    

def random_walk_for_hypergraph(R,
                               start_idx,
                               length,
                               lazy=True,
                               node_weight_func=None,
                               node2vec_q=None,
                               rand_seed=None,
                               Rcsr=None):
    """Generating a random walk with a specific length and from 
    a starting point 

    The input vertex matrix should be in CSC format (ensure to run
    `R.tocsc()` before feeding `R` to this function.
    """

    seq = [start_idx]       # set of hyper-nodes
    eseq = []               # set of hyper-edges

    if not(lazy) and (np.sum(R[:,start_idx])==0):
        print("Non-lazy random walk cannot start from an isolated vertex.")
        return None

    np.random.seed(rand_seed)
    randgen = np.random.sample

    if node2vec_q is not None:
        q = node2vec_q
        prev_idx = None    # previous (hyper)node

    if Rcsr is None:
        Rcsr = R.tocsr()

    v = start_idx
    for i in range(length-1):

        '''selecting edge e'''
        if node2vec_q is not None:
            e = node2vec_sample_edge(R, v, prev_idx, q, randgen)
            prev_idx = v   # update previous node
        else:
            v_edges = R[:,v].indices
            edge_weights = R[:,v].data   # this is an np.array
            eind = (edge_weights/edge_weights.sum()).cumsum().searchsorted(randgen())
            e = v_edges[eind]

        eseq += [e]

        '''selecting a node inside e'''
        row = np.float32(np.squeeze(Rcsr[e,:].toarray()))

        if not(lazy):
            row[v]=0

        # if no more remaining nodes, just finish the sampling
        if ~np.any(row>0):
            return seq, eseq

        if node_weight_func is None:
            e_nodes = np.where(row>0)[0]
            node_weights = row[row>0]
            node_weights = node_weights/node_weights.sum()
        else:
            # applying the node weight function before node selection
            node_weights = node_weight_func(row)
            
            if ~np.any(node_weights>0):
                return seq, eseq
            
            e_nodes = np.where(node_weights>0)[0]
            node_weights = node_weights[node_weights>0]

        CSW = node_weights.cumsum()
        if CSW[-1]<1.: CSW[-1]=1.
        nind = CSW.searchsorted(randgen())
        v = e_nodes[nind]

        seq += [v]

    return seq, eseq

    
def compute_multistep_transprob(P, source_inds, dest_inds, **kwargs):
    """Computing probability of multi-step transitions between two sets of nodes
    via a third intermediary set of nodes
    """

    interm_inds = kwargs.get('interm_inds', None)
    nstep = kwargs.get('nstep', 1)

    if interm_inds is None:
        # number of authors 
        msdb.crsr.execute('SELECT COUNT(*) FROM author;')
        nA = msdb.crsr.fetchone()[0]
        interm_inds = np.arange(nA)

    source_subP = P[source_inds,:]
    dest_subP = P[:,dest_inds]

    if nstep == 1:
        return source_subP[:,dest_inds]
    
    elif nstep==2:
        return source_subP[:,interm_inds] * dest_subP[interm_inds,:]
    
    elif nstep > 2:
        # for nstep=t, we need to have
        # P[source,A] * P[A,A]^t * P[A,dest] =
        # (((P[source,A] * P[A,A]) * P[A,A]) * ... ) * P[A,A] * P[A,inds]
        #               |------------------------------------|
        #                multiply for t times (preserve the order)
        #

        interm_subP = P[interm_inds,:][:,interm_inds]    #P[A,A]
        left_mat = source_subP[:,interm_inds] * interm_subP
        for t in range(1,nstep-2):
            left_mat = left_mat * interm_subP
        return left_mat * dest_subP[interm_inds,:]


def node2vec_sample_edge(R, curr_idx, prev_idx, q, randgen):
    """Sampling an edge in a node2vec style, starting from
    a current node `curr_idx` and given the previous node `prev_idx`
    with `p` and `q` the return and in-out parameters, respectively
    """

    N0 = R[:,curr_idx].indices
    
    # regular sampling in the first step
    if prev_idx is None:
        edge_weights = R[:,curr_idx].data   # this is an np.array
    else:
        # see which papers in N0 include previous node too (N0 intersect. N_{-1})
        edge_weights = np.ones(len(N0))
        N1 = R[:,prev_idx].indices
        edge_weights[np.isin(N0,N1)] = 1      # d_tx=1 in node2vec
        edge_weights[~np.isin(N0,N1)] = 1/q   # d_tx=2 in node2vec

    eind = (edge_weights/edge_weights.sum()).cumsum().searchsorted(randgen())
    e = N0[eind]
    
    return e