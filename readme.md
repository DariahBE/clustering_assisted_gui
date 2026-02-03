# Interactive cluster helper
CLARIAH-tool WP6.2

## Installation: 
1. Download repository from ()[]
2. Extract repository in a folder
3. Install a python3.12 virtual environment in that folder: `python3.12 -m venv .venv`
4. Activate the environment:
    WINDOWS PowerShell: `.venv/scripts/activate`
    LINUX Terminal: `source .venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`


## Usage: 
The tool requires a flat list of name entries you want to cluster, The order of the output matches the order of the input. All files are treated as if having only a single column, the filetypes: .txt, .csv and .tsv are supported. 

To use the tool, the notebook needs to run once from start to finish by clicking the 'run all' button. Once this has been completed, each individual step can be used in order to produce clusters of unseen data. Individual steps can be revisited as much as needed, however, downstream changes are only visible if all intermediate steps have been followed.
e.g.: all steps have ran 1,2,3,4,5 ==> user modifies step 2 ==> to reflect the made changes in the output; the user is required to perform steps 3 and 4 in succession.  

### Steps explained: 
#### step1: embeddings: 
Here the user should upload their data to the tool anc pick one of the available transformer models (or supply a custom model). Once all choices are made, the input list which consists of strings will be vectorized into a format which clustering algorithms can understand. A good first candidate is LABSE.

#### step2: dimension reduction: 
Embedders produce high-dimensional output, typically too much dimensions to be of any meaningful help for clustering. Here the user needs to pick a dimensionality reduction algorithm. A good first candidate is UMAP. 

#### step3: Dimensionality exploration: 
Optional step that visualizes the variance within each dimension. An optional sort parameter, sorts the datapoints in a single dimension from high to low. For the embeddings to be meaningful for your downstream clustering task, you'd want to see 'vertical groups' in your heatmap chart and 'block' in your linechart where datapoints follow a similar pattern for each cluster. On larger datasets, it becomes more difficult to achieve this effect. 

#### step4: Clustering:
There are a few clustering algorithms included that work out of the box. These algorithms will try to find groups of points that are closely together in the multidimensional space that was created in the previous step. These groups are called clusters - if a point cannot be assigned to a cluster (depends on the chosen algorithm and the scatter of each point), then it's marked as 'noise' with the label -1. After clustering a scatterplot in 2 or 3 dimensions is shown that displays the data - each cluster is labeled. The visualization is interactive and allows you to explore the clusterlabels and original data together. A second tab gives meta-statistics about the clustering process. 

#### step5: Noise extnsion: 
Unlabeled datapoints (noise) can be converted to a non-noise point by using one of two provided noise-reduction strategies. Both are based on KNN implementations. The first is a slow, iterative version that only looks at a single point at a time, this approach does not scale well and thus is better suited for smaller datasets. For larger dataset a faster version is implemented which assigns all noise points to the nearest cluster point in a single pass. Neither of the two implementations will create new cluster labels, however they'll reduce noise. 

#### step6: Cluster inspection and export: 
After completing the previous steps, the data is clustered. You can inspect a random cluster and find all instances in your dataset that share the same label, or look for a specific datapoint and find related mentions. You can choose wether or not you want to use your original labels, or noise-reduced labels. Once you completed the inspection and you agree with the clustering output, you can export the result to a new CSV. 
