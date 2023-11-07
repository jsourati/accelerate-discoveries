# Expert-Aware Discovery Acceleration
This package is implementation of an algorithm that incorporates distribution of experts and their experience for a better prediction of future scientific discoveries.

## Data
A few data files have been stored within `data/` to enable generating and evaluating discovery predictions for two example properties: thermoelectricity and COVID-19. Here's a brief description of the files:

* `data/*_vertex_matrix.npz`: vertex matrix of the literature hypergraph
* `data/*_mats.txt`: list of candidate materials (e.g., chemical compounds or approved drugs)
* `data/*_years.txt`: publication year of the articles included in our literature
* `data/*_groundtruth_discs.json`: ground-truth discoveries made by scientists in reality, which are grouped by months (for COVID-19) and years (for thermoelectricity)
    * For *Thermoelectricity*, the ground-truth discoveries are defined as materials that co-occurred with thermoelectricity keywords for the first time in the literature.
    * For *COVID-19*, the ground-truth discoveries are defined as materials (drugs) that have been involved in at least one clinical trial downloaded from ClinicalTrials.gov (the copied ground-truth is based on the list of trials retrieved on August 5, 2021).
* `data/*_model_*`: pre-trained word2vec sample model to be used for generating predictions

## An Example (Thermoelectricity)
Data necessary for running our expert-aware discovery prediction algorithm on an example property, i.e., "thermoelectricity", is included in `data/`. 
Here are the main steps:

### Forming the Literature Hypergraph:
The first step is to create or load the vertex matrix of our literature hypergraph. For thermoelectricity, this matrix is already prepared and stored as 
`data/thrm_vertex_matrix.npz`. We also need a list of materials, and a list of names for the properties at hand:
```
from scipy import sparse
import numpy as np
import literature

R = sparse.load_npz("data/thrm_vertex_matrix.npz")
mats = np.array(open("data/thrm_mats.txt", "r").read().splitlines())
props = ["thermoelectric"]
```
We have also included a file that includes publication year of the papers that we considered in our vertex matrix, which can be used to limit our focus to 
articles that are published in a certain range:
```
yrs = np.loadtxt('data/thrm_years.txt')
# We'll consider papers that are published in range [1996, 2000]
R = R[(yrs>=1996)*(yrs<=2000),:]
```
Using these materials, we can form the literature hypergraph:
```
h = literature.hypergraph(R, mats, props)
```

### Sampling Random Walk Sequences
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

### Training Word Embedding Model
The sequences generated from the random walk process could then be used to train an embedding model over the graph's nodes (in a [deepwalk fashion](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)). However, we won't need the author nodes for the task of discovery prediction. Hence, before moving forward we prune our random-walk sequences by removing the author nodes and saving the resulting sequences into a new file:
```
import utils 

with open("rw_seqs.txt", "w") as file:
    for i in range(100):
        rw_seqs = h.random_walk(length, size, start_inds=prop_ind, alpha=2, rand_seed=i)[0][0]    # non-uniform sampling (alpha=1)
        file.write(rw_seqs+'\n')

seqs = open("rw_seqs.txt").read().splitlines()                              # reading the sequences
seqs_noauthors = utils.remove_authors_from_RW(seqs)                         # removing the author nodes
open("rw_seqs_noauthors.txt", "w").write("\n".join(seqs_noauthors)+"\n")    # saving the pruned sequences
```

Now, we are ready to build our word embedding model. The default parameters are saved within file `default_params.json`. In order to use different values for these parameters, one can either change the values inside this file or input different values when initiating `embedding.dww2v`:
```
seqs_noauthor_path = "rw_seqs_noauthors.txt"

import embedding

embed = embedding.dww2v(seqs_noauthor_path, workers=20)     # initiating deepwalk model with a different value for parameter workers
embed.build_model()
embed.train()
```
Once the training is done, the word embedding model will be accessible through `embed.model`. We have also included a pretrained sample model for thermoelectricity among data files in `data/`. We, can load it into our embedding object: `embed.load_model("data/thrm_model_1996_2000")`.


### Inferring Materials with the Targeted Property
The (pre)trained word embedding models will be used to rank the candidate materials based on their similarity to the property (in this example, thermoelectricity). 
```
sims,_,reordered_mats = embed.similarities(['thermoelectric'], mats, return_nan=False)
```

**NOTE FOR ELECTROCHEMICAL PROPERTIES:** Above command computes similarity between *all* materials and the property. However, when working with electrochemical properties (e.g., thermoelectricity), first we have to discard materials that have been already co-occurred (i.e., studied) with the targeted property. For instance, for prediction year 2001, we'll have:
```
full_R = sparse.load_npz("data/thrm_vertex_matrix.npz")
subgraph_R = full_R[yrs<=2000]
studied_mats = mats[np.asarray(np.sum(subgraph_R[:,h.nA:-1].multiply(subgraph_R[:,-1]), axis=0)>0)[0,:]]
candidate_mats = mats[~np.isin(mats,studied_mats)]
```

The prediction will be done over these candidate materials:
```
sims,_,reordered_mats = embed.similarities(['thermoelectric'], candidate_mats, return_nan=False)

# reporting 50 materials with highest likelihood of being thermoelectric
preds = reordered_mats[np.argsort(-sims[0,:])][:50]
```

## Evaluating Inferred Materials
The evaluation will be done based on the ground-truth discoveries saved as `data/thrm_groundtruth_discs.json`. For instance, we can compute the year-wise cumulative precision of the predicted materials as follows:
```
import json

gt_discs = json.load(open("data/thrm_groundtruth_discs.json","r")) 
yearwise_precs = [np.isin(preds,gt_discs[str(x)]).sum()/len(preds) for x in range(2001,2019)]
np.cumsum(yearwise_precs)
```
