# Expert-Aware Discovery Acceleration
This package is implementation of an algorithm that incorporates distribution of experts and their experience for a better prediction of future scientific discoveries.

## An Example
Data necessary for running our expert-aware discovery prediction algorithm on an example property, i.e., "thermoelectricity", is included in `data/`. 
Here are the main steps:

* **Forming the Literature Hypergraph:**
The first step is to create or load the vertex matrix of our literature hypergraph. For thermoelectricity, this matrix is already prepared and stored as 
`data/thrm_vertex_matrix.npz`. We also need a list of materials, and a list of names for the properties at hand:
```
from scipy import sparse
import numpy as np
import literature

R = sparse.load_npz("data/thrm_vertex_matrix.npz")
mats = np.array(open("data/thrm_mats.txt", "r").read().splitlines)
props = ["thermoelectric"]
```
We have also included a file that includes publication year of the papers that we considered in our vertex matrix, which can be used to limit our focus to 
articles that are published in a certain range:
```
yrs = np.loadtxt('data/thrm_years.txt')
# We'll consider papers that are published in range [1996, 2000]
R = = R[(yrs>=1996)*(yrs<=2000),:]
```
Using these materials, we can form the literature hypergraph:
```
h = literature.hypergraph(R, mats, props)
```

* Sampling Random Walk Sequences
Next step is to let a random walker sample from the resulting hypergraph. Before running the random walker, here are some parameters to set:
```
length = 20                 # length of the walk
size = 1                    # number of the walk
prop_ind = R.shape[1]-1     # column index of the property as the starting node 
```
Then, we can simply call the `random_walk` method of the hypergraph. For uniform sampling:
```
h.random_walk(length, size, start_inds=prop_ind, rand_seed=0)    # uniform sampling

# resulting in the following output: 
# (the first array is the sequence of selected nodes; the second array is the selected papers along the walk):
# ---------------------
# (['thermoelectric a_1244326 a_1084770 a_1085357 CoCrFeMnNi a_281555 a_1076970 CSi a_10764 Al2O3
# K2O a_1672448 CaF2 a_460834 BaF2 a_638548 a_1287239 a_955446 a_955445 a_955447'],
#  ['962469 1191497 746280 1191497 1421491 734403 1115449 132804 46832 1194889 1400463 1400463 23
# 2314 232314 894012 1035899 1035899 615755 1075096'])
```
And for non-uniform sampling:
```
h.random_walk(length, size, start_inds=prop_ind, alpha=1, rand_seed=1)    # non-uniform sampling (alpha=1)

# resulting in the following output:  
# (the first array is the sequence of selected nodes; the second array is the selected papers along the walk):
# ---------------------
# (['thermoelectric a_1201042 a_1172586 a_667 a_481602 CO2 a_811620 CO2 CH4 CH2 a_148079 KMnO4 CF
# 2 Al2O3 a_223689 AsU a_618201 Fe2O3 a_67672 CdS'],
#  ['866173 866173 835252 265095 245713 503047 503047 34062 473369 132162 1064758 360666 1170038
# 129996 129885 337769 337769 1066194 1357414'])
```
