#!/usr/bin/env python
# coding: utf-8

# In[292]:


get_ipython().run_line_magic('matplotlib', 'inline')
import htree.logger as logger
import htree.conf as conf
from htree.tree_collections import MultiTree, Tree

logger.set_logger(False, log_dir="/Users/smirarab/tmp")


# In[108]:


# This is a hack for a missing function, enabling us to embed given a distance matrix directly instead of trees
from htree import utils  as utils
import htree.embedding as embedding
import os
import pickle

def embed_dist(
        dim: int, dist, labels,
        geometry: str = 'hyperbolic',
        **kwargs
    ):
    if dim is None:
        raise ValueError("Parameter 'dim' is required.")
    # Parameter defaults
    defaults = {
        'precise_opt': conf.ENABLE_ACCURATE_OPTIMIZATION, 'epochs': conf.TOTAL_EPOCHS,
        'lr_init': conf.INITIAL_LEARNING_RATE, 'dist_cutoff': conf.MAX_RANGE,
        'export_video': conf.ENABLE_VIDEO_EXPORT, 'save_mode': conf.ENABLE_SAVE_MODE,
        'scale_fn': None, 'lr_fn': None, 'weight_exp_fn': None, 'curvature': None,
    }
    params = {k: kwargs.get(k, v) for k, v in defaults.items()}
    params['save_mode'] |= params['export_video']
    params['export_video'] &= params['precise_opt']
    is_hyperbolic = geometry == 'hyperbolic'
    try:
        dist_matrix, curvature = dist, None
        # Hyperbolic: scale distances and compute curvature
        if is_hyperbolic:
            if params['curvature'] is not None and params['curvature'] >= 0:
                print("Wrong input curvature. It has to be negative.")
                return None
            if params['curvature'] is not None:
                curvature, params['scale_fn'] = params['curvature'], lambda x1, x2, x3: False
                scale = np.sqrt(-curvature)
            else:
                scale = params['dist_cutoff'] / torch.max(refdist)
                curvature = -(scale ** 2)
            dist_matrix = dist_matrix * scale
        # Naive embedding initialization
        points = utils.naive_embedding(dist_matrix, dim, geometry=geometry)
        # Precise optimization refinement
        if params['precise_opt']:
            opt_result = utils.precise_embedding(
                dist_matrix, dim, geometry=geometry, init_pts=points, **params)
            points, opt_scale = (opt_result, 1) if not is_hyperbolic else opt_result
            curvature = curvature * opt_scale ** 2 if is_hyperbolic else None
        # Construct embedding object
        result = (embedding.LoidEmbedding(points=points, labels=labels, curvature=curvature)
                  if is_hyperbolic else embedding.EuclideanEmbedding(points=points, labels=labels))
    except Exception as e:
        raise
    # Save embedding to timestamped directory
    out_dir = os.path.join(conf.OUTPUT_DIRECTORY, "test")
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"{geometry}_embedding_{dim}d.pkl")
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    except (IOError, pickle.PicklingError) as e:
        raise
    return result


# In[314]:


# Read gene trees
mt = MultiTree("Fig3-genetrees.tre")


# In[318]:


# Normalize gene tree embeddings to get rid or overal rate variation
# Skip to do unscaled ...

rates = mt.normalize()

#Visualize rates
import matplotlib.pyplot as plt
plt.hist(rates, bins=40) # Convert to numpy array
plt.title('Histogram of relative gene rates for the plant dataset')

plt.savefig("plantrates.pdf", format='pdf', bbox_inches='tight') #

plt.show()


# In[ ]:


# Align the species trees to gene trees
import torch
from htree.procrustes import HyperbolicProcrustes
import numpy as np

st = Tree("astral-Fig3-rooted.scored.tre")

stdiam = st.diameter()

# Distance between two sets of embeddings
def compute_dist(ref,tar):
    return ref.poincare_distance(
        ref.to_poincare(ref.points),
        tar.to_poincare(tar.points[:, [tar._labels.index(label) for label in ref._labels]])).tolist()[0]

# Infer a median tree and compared to ASTRAL
def spterr(refs,d):
    m,l = refs.distance_matrix()
    with open('dist%d.txt' %d,'w') as f:
        f.write("%d\n%s\n" %(len(l), "\n".join("%s %s" %(l[i],"\t".join(f"{a:.6g}" for a in m[i].tolist())) for i in range(0,len(l)))))
    subprocess.run(["/Users/smirarab/miniforge3/bin/fastme", "-i", "dist.txt"],capture_output=True,text=True,check=True)
    result = subprocess.run(["compareTrees.missingBranch",  "astral-Fig3-rooted.scored.tre", "dist.txt_fastme_tree.nwk"],capture_output=True,text=True,check=True)
    return result.stdout.split()[2],torch.max(m)

meanres = {}
for d in range (2,21):
    meanres[d] = []

    # Embed gene trees and the species tree
    me = mt.embed(dim=d, geometry='hyperbolic',precise_opt=True)
    se = st.embed(dim=d, geometry='hyperbolic',precise_opt=True,curvature=me[0].curvature)

    refs = me.reference_embedding(func = torch.nanmean,precise_opt=True)
    aggerror, diam = spterr(refs,d) 
    sd = [(aligned,unaligned) for (aligned,unaligned) in zip(compute_dist(refs, HyperbolicProcrustes(se,refs).map(se)),compute_dist(refs,se))]
    meanres[d].append(("mean", np.mean([sdi[0] for sdi in sd]),np.mean([sdi[1] for sdi in sd]),aggerror))

    refs = me.reference_embedding(func = torch.nanmedian,precise_opt=True)
    aggerror, diam  = spterr(refs,d) 
    sd = [(aligned,unaligned) for (aligned,unaligned) in zip(compute_dist(refs, HyperbolicProcrustes(se,refs).map(se)),compute_dist(refs,se))]
    meanres[d].append(("median", np.mean([sdi[0] for sdi in sd]),np.mean([sdi[1] for sdi in sd]),aggerror))

    # A Bit of hacking needed for the Gaussian weighting method. 
    refdist, C, l = mt.distance_matrix(method = "fp")
    aggerror, diam  = spterr(refs,d) 
    refs = embed_dist(dim=d, dist=refdist, labels=l, geometry='hyperbolic',precise_opt=True,curvature=me[0].curvature)
    sd = [(aligned,unaligned) for (aligned,unaligned) in zip(compute_dist(refs, HyperbolicProcrustes(se,refs).map(se)),compute_dist(refs,se))]
    meanres[d].append(("FP", np.mean([sdi[0] for sdi in sd]),np.mean([sdi[1] for sdi in sd]),aggerror))

    print(meanres)


# In[317]:


with open('gt-dimensions-scaled-fullepochs.txt','w') as f:
    f.write("Dim MeanMethod DistortionUnaligned DistortionAligned STERR\n%s" %("\n".join("%d %s %f %f %s" %(d,t[0],t[2],t[1],t[3]) for d in range(2,21) for t in meanres[d] )))


# In[ ]:


# Move ahead with dimension 12 
d = 12
# Embed gene trees
me = mt.embed(dim=d, geometry='hyperbolic',precise_opt=True)
# Read and embed the species trees
st = Tree("astral-Fig3-rooted.scored.tre")
se = st.embed(dim=d, geometry='hyperbolic',precise_opt=True,curvature=me[0].curvature)


# In[321]:


# Align the gene tree embeddings
import torch
me.align(func = torch.nanmean, precise_opt = 'accurate')


# In[322]:


# Save the unaligned gene tree embeddings
me.save("1KPalignedgenetrees.aligned.embed.pickle")


# In[116]:


from htree import procrustes
def align(me, reference_embedding, **kwargs) -> None:
    """
    Aligns all embeddings by averaging their distance matrices and adjusting
    each embedding to match the reference embedding.
    """
    newembds = []
    if not me.embeddings:
        me._log_info("No embeddings to align.")
        return

    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'func'}
    if me.curvature < 0:
        for i, embedding in enumerate(me.embeddings):
            model = procrustes.HyperbolicProcrustes(embedding, reference_embedding,**filtered_kwargs)
            newembds.append( model.map(embedding) )
    else:
        for i, embedding in enumerate(me.embeddings):
            model = procrustes.EuclideanProcrustes(embedding, reference_embedding,**filtered_kwargs)
            newembds.append( model.map(embedding) )


# In[118]:


align(me,refs, precise_opt = 'accurate')


# In[119]:


# Save the aligned gene tree embeddings
me.save("1KPalignedgenetrees.embed.pickle")



# Read and embed the species trees
import subprocess
st = Tree("astral-Fig3-rooted.scored.tre")
std,ld = st.distance_matrix()
res = []
distor = []
for d in range(2,21):
    se = st.embed(dim=d, geometry='hyperbolic',precise_opt=True)
    m,l = se.distance_matrix()
    with open('se.txt','w') as f:
        f.write("%d\n%s\n" %(len(l), "\n".join("%s %s" %(l[i],"\t".join(f"{a:.6g}" for a in m[i].tolist())) for i in range(0,len(l)))))
    subprocess.run(["/Users/smirarab/miniforge3/bin/fastme", "-i", "se.txt"],capture_output=True,text=True,check=True)
    result = subprocess.run(["compareTrees.missingBranch",  "astral-Fig3-rooted.scored.tre", "se.txt_fastme_tree.nwk"],capture_output=True,text=True,check=True)
    res.append(result.stdout.split()[2])
    distor.append(float(sum(sum((m-std)**2))))
    print(distor[-1],res[-1])
with open('st-dimensions.txt','w') as f:
    f.write("Dim Distortion STERR\n%s" %("\n".join("%d %f %s" %(d,distor[d-2],res[d-2]) for d in range(2,21))))


# In[278]:


with open('st-dimensions.txt','w') as f:
    f.write("Dim Distortion STERR\n%s" %("\n".join("%d %f %s" %(d,distor[d-2],res[d-2]) for d in range(2,21))))

