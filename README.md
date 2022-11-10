# QGCN
> QGCN method for graph classification: https://arxiv.org/abs/2104.06750


## Installation
> required packages:
> - scipy~=1.8.0
> - pandas~=1.4.2
> - networkx~=2.8.3
> - numpy~=1.22.3
> - torch~=1.11.0
> - scikit-learn~=1.1.1
> - bokeh~=2.4.2
> - matplotlib~=3.5.1
> - bitstring~=3.1.9
> - python-louvain~=0.16
> - graph-measures~=0.1.44

You can download the package by the command:
```
pip install QGCN
```

<br />

[//]: # (## How to use )
## Graph representing
To use this package you will need to provide the following files as input:

* **_Graphs csv file:_** files that contain the graphs for input and their labels.
  The format of the file is flexible, but it must contain headers for any column, and there must be a column provided for:
  - graph id
  - source node id
  - destination node id
  - label id (every graph id can be attached to only one label)
- **_External data file:_** external data for every node (Optional)
    The format of this file is also flexible, but it must contain headers for any column, and there must be a column provided for:
    **note!! every node must get a value**
    - graph id
    - node id
    - column for every external feature (if the value is not numeric then it can be handled with embeddings)  

<br />

Example for such files:
* **_graph csv file:_**

```csv
g_id,src,dst,label
6678,_1,_2,i
6678,_1,_3,i
6678,_2,_4,i
6678,_3,_5,i
```

* **_External data file:_**
```csv
g_id,node,charge,chem,symbol,x,y
6678,_1,0,1,C,4.5981,-0.25
6678,_2,0,1,C,5.4641,0.25
6678,_3,0,1,C,3.7321,0.25
6678,_4,0,1,C,6.3301,-0.25
```

<br />

## Parameters passing

After creating these file, you should define the parameters of the model. This can be done with a json file, or with data classes.
The parameters split to 4 groups:
>* **_graphs_data:_**
>  * file_path - the path to the graph csv file (with the edges and labels for each graph)
>  * graph_col - the name of the column with the graph id
>  * src_col - the name of the column with the source node of the edge
>  * dst_col - the name of the column with the target node of the edge
>  * label_col - the name of the column with the label of the graph
>  * directed - indicates if the graph is directed (gets True/False)
>  * features - list of topologic features which will be calculated to the nodes.
>    * The options are - ["DEG", "CENTRALITY", "BFS"]
>    * You can read more about it here >>
>  * adjacency_norm - the norm which will be used (get examples)
>    * The options are - "NORM_REDUCED", "NORM_REDUCED_SYMMETRIC", "IDENTITY", "RAW_FORM"
>  * standardization - the standardization which will be used
>    * The options are - "zscore", "min_max", "scale"
>
>
>* **_external:_**
>  * file_path - the path to the external data csv file (with other node features)
>  * graph_col - the name of the column with the graph id
>  * node_col - the name of the column with the node id
>  * embeddings - a list with the names of the embeddings features of the nodes
>  * continuous - a list with the names of the continuous features of the nodes
>
>
>* **_model:_**
>  * label_type - 'binary' if the predication in binary, 'multi' else
>  * num_classes - number of label types
>  * use_embeddings - if the model should use the embeddings features (gets True/False)
>  * embeddings_dim - a list with the dimensions of the embeddings features
>  * activation - the activation function which will be used. 
>    * Notice that the activation function will be combined with SRSS function. 
>    * The options are - "relu_", "tanh_", "sigmoid_", "srss_"
>  * dropout - the dropout rate of the model
>  * lr - the learning rate of the model
>  * optimizer - the optimizer of the model
>    * The options are - "ADAM_", "SGD_"
>  * L2_regularization - the L2_regularization rate of the model 
>  * GCN_layers - an array with dictionaries for each layer. 
>    * for example: [ <br>
>            { "in_dim": "None", "out_dim": 100 }, <br>
>            { "in_dim": 100, "out_dim": 50 }, <br>
>            { "in_dim": 50, "out_dim": 25 } <br>
>        ]
>
>
>* **_activator:_**
>  * epochs - the epochs number of the model
>  * batch_size - the size of each batch
>  * loss_func - the loss function which will be used
>  * train - percentage of the data which will used for train
>  * dev - percentage of the data which will used for dev
>  * test - percentage of the data which will used for test

<br/>

* Example json file:
  - (Notice that if an external file is not provided, you should put the associated parameters as None.)
  - you can find complete params files [here](https://github.com/louzounlab/QGCN/tree/main/example/params).
```json
{
    "dataset_name": "DataSetName",

    "external": {
       -- external params here -- 
    },

    "graphs_data": {
        -- graphs_data here --
    },

    "model": {
        -- model params here --
    },

    "activator": {
        -- activator params here -- 
    }
}
```
<br />

* Example dataclass objects:
  * The dataclasses default values are [here](https://github.com/louzounlab/QGCN/blob/main/src/QGCN/params.py).
```python
from QGCN.params import GraphsDataParams, ExternalParams, ModelParams, ActivatorParams 

external_params = ExternalParams(file_path="./data/Mutagenicity_external_data_all.csv",
                          embeddings=["chem"],
                          continuous=[])

graphs_data_params = GraphsDataParams(file_path="../src/QGCN/data/Mutagenicity_all.csv",
                               standardization="min_max")

model_params = ModelParams(label_type="binary",
                    use_embeddings="True",
                    embeddings_dim=[10],
                    activation="srss_",
                    GCN_layers=[
                        {"in_dim": "None", "out_dim": 250},
                        {"in_dim": 250, "out_dim": 100}])

activator_params = ActivatorParams(epochs=100)
```

<br />

## Executing the model
Once you have these files, you can use the QGCNModel from QGCN.activator with the path to the parameters file or the dataclass objects:
```python
from QGCN.activator import QGCNModel, QGCNDataSet

qgcn_model = QGCNModel(dataset_name="Aids", params_file="params.json")
qgcn_model.train()
```
```python
from torch.utils.data import DataLoader
from QGCN.params import GraphsDataParams, ExternalParams, ModelParams, ActivatorParams 
from QGCN.activator import QGCNModel, QGCNDataSet

# sets the parameters of the dataset:
graphs_data = GraphsDataParams(file_path="./data/data_all.csv",
                               standardization="min_max")
external = ExternalParams(file_path="./data/external_data_all.csv",
                          graph_col="g_id", node_col="node",
                          embeddings=["chem"], continuous=[])


# sets the parameters of the model:
model = ModelParams(label_type="binary", num_classes=2, use_embeddings="True", embeddings_dim=[10],
                    activation="srss_", dropout=0.2, lr=0.005, optimizer="ADAM_", L2_regularization=0.005, f="x1_x0")
activator = ActivatorParams(epochs=100)

qgcn_model = QGCNModel("Mutagen", graphs_data, external, model, activator)
qgcn_model.train(should_print=True)
```

<br />

## Links
The datasets can be download here: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets . Notice you will have to change their format to ours. You can see an example data here (gitHub link) the conventor in datasets -> change_data_format.py
Mail address for more information: 123shovalf@gmail.com
