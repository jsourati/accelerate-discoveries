import os
import sys
import pdb
import json
import random
import logging
import pymysql
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy import sparse
from collections import deque
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count


class hypergraph(object):

    def __init__(self, R, mats, props, authornames=None):
        '''Make sure to feed the weight matrix in a format such that 
        the first batch of columns (nA) corresponds to authors, the second
        batch (nM) corresponds to the pool of materials and the third batch (nP)
        corresponds to properties
        '''
        
        self.R = R
        self.mats = np.array(mats)
        self.props = np.array(props)
        self.nM = len(mats)
        self.nP = len(props)
        self.nA = R.shape[1] - self.nM - self.nP
        if authornames is not None:
            assert len(authornames)==self.nA, "Number of author names should match " + \
                "the number of clumns in the node matrix."
            if (np.isin(authornames, self.mats).sum() +
                np.isin(authornames, self.props).sum()) > 0:
                print('WARNING: there is an ambiguity in node naming.')
            self.authornames = authornames
        else:
            self.authornames = np.array(['a_{}'.format(i) for i in range(self.nA)])

        self.nodenames = np.concatenate([self.authornames, self.mats, self.props])

            
    def get_csr_mat(self):
        self.Rcsr = self.R.tocsr()
        
    def node_to_type(self,idx):
        if idx < self.nA:
            return 'author'
        elif self.nA <= idx < self.nA+self.nM:
            return 'material'
        elif self.nA+self.nM <= idx < self.nA+self.nM+self.nP:
            return 'property'
        else:
            raise ValueError('Given node index not in a valid range.')

    def node_to_name(self, idx):

        if 0 <= idx < self.nA+self.nM+self.nP:
            return self.nodenames[idx]
        else:
            raise ValueError('Given node index not in a valid range.')

    def name_to_node(self, name):
        if name in self.nodenames:
            return np.where(self.nodenames==name)[0][0]
        else:
            return np.nan

    def search_name(self, name):
        return [x for x in self.nodenames if name in x]
        
            
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
            names =  [self.node_to_name(x) for x in nodes_inds]
            types = [self.node_to_type(x) for x in nodes_inds]
            return {'authors': [self.node_to_name(x) for x in nodes_inds
                               if self.node_to_type(x)=='author'],
                    'materials': [self.node_to_name(x) for x in nodes_inds
                                  if self.node_to_type(x)=='material'],
                    'properties': [self.node_to_name(x) for x in nodes_inds
                               if self.node_to_type(x)=='property']}
        else:
            return nodes_inds

        
    def node_to_papers(self, node_identifiers):
        '''Returning all papers that include a given node
        '''


        # if multiple identifiers are given, papers that contain all the
        # nodes are returned (intersection of inidividual papers)
        if isinstance(node_identifiers, (list, tuple, np.ndarray)):
            for i, idx_or_name in enumerate(node_identifiers):
                if i==0:
                    papers = self.node_to_papers(idx_or_name)
                else:
                    papers = papers[np.isin(papers,self.node_to_papers(idx_or_name))]
                
        else:
            idx_or_name = node_identifiers
            if isinstance(idx_or_name, str):
                idx = self.name_to_node(idx_or_name)
            else:
                idx = idx_or_name
            papers = self.R[:,idx].nonzero()[0]
        
        return papers

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
        workers = kwargs.get('workers', 1)

        
        # setting the initial node index    
        if start_inds is None:
            n = self.R.shape[1]
            init_idx = np.random.randint(0,n,size) if rand_seed is None else \
                np.random.RandomState(rand_seed).randint(0,n,size)
        elif isinstance(start_inds, (list,np.ndarray)):
            # randomly choose one of them
            rand_idx = np.random.randint(0,len(start_inds),size) if rand_seed is None \
                else np.random.RandomState(rand_seed).randint(0,len(start_inds),size)
            init_idx = [start_inds[x] for x in rand_idx]
        elif np.isscalar(start_inds):
            if start_inds-int(start_inds) != 0:
                raise ValueError("The starting index in a random walk should be " +\
                                 "a positive integer (not a float like {}).".format(start_inds))
            init_idx = np.ones(size,dtype=int) * int(start_inds)
            
            
        if rand_seed is None:
            rand_seeds = [None]*size
        else:
            rand_seeds = rand_seed + np.arange(size)

        Rcsr = self.Rcsr if hasattr(self,'Rcsr') else None
            
        ''' Iteratively generate random walk sequences'''
        if workers==1:
            
            nseqs = []
            eseqs = []
            nlines=0
            
            # sequential random-walk
            for i in range(size):
                nseq, eseq = random_walk_for_hypergraph(self,
                                                        init_idx[i],
                                                        length,
                                                        lazy=False,
                                                        alpha=alpha,
                                                        node2vec_q=node2vec_q,
                                                        rand_seed=rand_seeds[i])

                eseqs += [' '.join([str(x) for x in eseq])]

                # parsing the hyper nodes
                toks = [self.node_to_name(s) for s in nseq]
                nseq = ' '.join(toks)
                nseqs += [nseq]

                if not(i%500) and i>0:
                    if nseq_file_path is not None:
                        with open(nseq_file_path, 'a') as tfile:
                            tfile.write('\n'.join(nseqs[i-500:i])+'\n')
                            nlines = i
                    if eseq_file_path:
                        with open(eseq_file_path, 'a') as tfile:
                            tfile.write('\n'.join(eseqs[i-500:i])+'\n')
                            nlines = i

            if nseq_file_path is not None:
                with open(nseq_file_path, 'a') as f:
                    f.write('\n'.join(nseqs[nlines:])+'\n')
            if eseq_file_path is not None:
                with open(eseq_file_path, 'a') as f:
                    f.write('\n'.join(eseqs[nlines:])+'\n')
                    
        elif workers>1:
            
            # parallel random-walk sequence generation
            tqdm_list = tqdm(range(size), position=0, leave=True)
            with Parallel(n_jobs=workers, backend="multiprocessing") as parallel_processor:
                res = parallel_processor(delayed(random_walk_for_hypergraph)(
                    self,
                    init_idx[i],
                    length,
                    lazy=False,
                    alpha=alpha,
                    node2vec_q=node2vec_q,
                    rand_seed=rand_seeds[i]) for i in tqdm_list)
            
            nseqs, eseqs = list(zip(*res))
            eseqs = [' '.join([str(x) for x in eseq]) for eseq in eseqs]
            nseqs = [' '.join([self.node_to_name(x) for x in nseq]) for nseq in nseqs]

            if nseq_file_path is not None:
                with open(nseq_file_path, 'w') as f:
                    f.write('\n'.join(nseqs)+'\n')
            if eseq_file_path is not None:
                with open(eseq_file_path, 'w') as f:
                    f.write('\n'.join(eseqs)+'\n')

        return nseqs, eseqs

    

def random_walk_for_hypergraph(h,
                               start_idx,
                               length,
                               lazy=True,
                               alpha=None,
                               node2vec_q=None,
                               rand_seed=None):
    """Generating a random walk with a specific length and from 
    a starting point 

    The input vertex matrix should be in CSC format (ensure to run
    `R.tocsc()` before feeding `R` to this function.
    """

    seq = [start_idx]       # set of hyper-nodes
    eseq = []               # set of hyper-edges

    if not(lazy) and (np.sum(h.R[:,start_idx])==0):
        print("Non-lazy random walk cannot start from an isolated vertex.")
        return None

    np.random.seed(rand_seed)
    randgen = np.random.sample
    
    if not hasattr(h, 'Rcsr'):
        h.get_csr_mat()

    # whether alpha-sampling is being used
    if alpha is None:
        # uniform sampling
        node_weight_func = None
    elif np.isscalar(alpha):
        # alpha-modified sampling
        def node_weight_func(data):
            return alpha_modify_dist(alpha, data, h.nA, h.nM)
    
    # whether node2vec type of sampling is being used
    if node2vec_q is not None:
        q = node2vec_q
        prev_idx = None    # previous (hyper)node

    v = start_idx
    for i in range(length-1):

        '''selecting edge e'''
        if node2vec_q is not None:
            e = node2vec_sample_edge(h.R, v, prev_idx, q, randgen)
            prev_idx = v   # update previous node
        else:
            v_edges = h.R[:,v].indices
            edge_weights = h.R[:,v].data   # this is an np.array
            eind = (edge_weights/edge_weights.sum()).cumsum().searchsorted(randgen())
            e = v_edges[eind]

        eseq += [e]

        '''selecting a node inside e'''
        row = np.float32(np.squeeze(h.Rcsr[e,:].toarray()))

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

    
def alpha_modify_dist(alpha, hotvec, nA, nM):
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
        hotvec[:nA] = 0

    sum_AP = np.sum(hotvec[:nA]) + np.sum(hotvec[nA+nM:])
    sum_M = np.sum(hotvec[nA:nA+nM])

    if sum_AP>0:
        hotvec[:nA] = hotvec[:nA] / sum_AP
        hotvec[nA+nM:] = hotvec[nA+nM:] / sum_AP
    if sum_M>0:
        hotvec[nA:nA+nM] = alpha*hotvec[nA:nA+nM] / sum_M


    return hotvec/np.sum(hotvec)


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
