
##VECTORS: 
vectors = None   #whenever user clicks the 'Get embeddings' button, the resulting embeddings are hoisted into here.
names = None
def receive_embeddings(emb, inputlist):
    global vectors, names
    vectors = emb
    names = inputlist

def get_input_list():
    global names
    return names


## DIMENSIONALITY REDUCTION: 
reduced = None 
def receive_reduced(red):
    global reduced
    reduced = red
def fetch_vectors(): 
    global vectors
    return vectors


## CLUSTERING: 
clusters = None 
def receive_vectorlabels(labels):
    global clusters
    clusters = labels
def get_reduced(): 
    global reduced
    return reduced

## CLUSTER INSPECTION: 
def get_cluster_labels(): 
    global clusters
    return clusters


## DENOISE: 
noise_labels = None
noise_source = None

def receive_denoised_results(labels, source): 
    global noise_labels, noise_source
    noise_labels = labels
    noise_source = source

def get_denoised_labels(): 
    global noise_labels
    return noise_labels

def get_denoised_sources(): 
    global noise_source 
    return noise_source