# QGCN
QGCN method for graph classification: https://arxiv.org/abs/2104.06750

## Installation
required packages:
- scipy~=1.8.0
- pandas~=1.4.2
- networkx~=2.8.3
- numpy~=1.22.3
- torch~=1.11.0
- scikit-learn~=1.1.1
- bokeh~=2.4.2
- matplotlib~=3.5.1
- bitstring~=3.1.9
- python-louvain~=0.16
- graph-measures~=0.1.44

You can download the package by the command:
```
pip install QGCN
```

## How to use 
### Graph representing
To use this package you will need to provide the following files as input:

* Graphs csv file - files that contain the graphs for input and their labels.
  The format of the file is flexible, but it must contain headers for any column, and there must be a column provided for:
  - graph id
  - source node id
  - destination node id
  - label id (every graph id can be attached to only one label)
- External data file - external data for every node (Optional)
    The format of this file is also flexible, but it must contain headers for any column, and there must be a column provided for:
    **note!! every node must get a value**
    - graph id
    - node id
    - column for every external feature (if the value is not numeric then it can be handled with embeddings)

Example for such files: <br>
graph csv file: 
```csv
g_id,src,dst,label
6678,_1,_2,i
6678,_1,_3,i
6678,_2,_4,i
6678,_3,_5,i
```

External data file:
```csv
g_id,node,charge,chem,symbol,x,y
6678,_1,0,1,C,4.5981,-0.25
6678,_2,0,1,C,5.4641,0.25
6678,_3,0,1,C,3.7321,0.25
6678,_4,0,1,C,6.3301,-0.25
```

### Parameters passing

After creating these file, you should define the parameters of the model. This can be done with a json file, or with data classes:
* Example json file:
  - (Notice that if an external file is not provided, you should put the associated parameters as None.)
```json
{
    "dataset_name": "Aids",

    "external": {
        "file_path": "./data/AIDS_external_data_all.csv",
        "graph_col": "g_id",
        "node_col": "node",
        "embeddings": ["chem", "symbol"],
        "continuous": ["charge", "x", "y"]
    },

    "graphs_data": {
        "file_path": "./data/AIDS_all.csv",
        "graph_col": "g_id",
        "src_col": "src",
        "dst_col": "dst",
        "label_col": "label",
        "directed": "False",
        "features": ["DEG", "CENTRALITY", "BFS"],
        "adjacency_norm": "NORM_REDUCED",
        "percentage": 1,
        "standardization": "zscore"
    },

    "model": {
        "label_type": "binary",
        "num_classes": 2,
        "use_embeddings": "True",
        "embeddings_dim": [10, 10],
        "activation": "relu_",
        "dropout": 0,
        "lr": 1e-3,
        "optimizer": "ADAM_",
        "L2_regularization": 0,
        "f": "c_x0",
        "GCN_layers": [
            { "in_dim": "None", "out_dim": 100 },
            { "in_dim": 100, "out_dim": 50 },
            { "in_dim": 50, "out_dim": 25 }
        ]
    },

    "activator": {
        "epochs": 3,
        "batch_size": 128,
        "loss_func": "binary_cross_entropy_with_logits_",
        "train": 0.3467,
        "dev": 0.1153,
        "test": 0.538
    }
}
```
* Example dataclass objects:
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

### Executing the model
Once you have these files, you can use the QGCNModel from QGCN.activator with the path to the parameters file or the dataclass objects:
```python
from torch.utils.data import DataLoader
from QGCN.params import GraphsDataParams, ExternalParams, ModelParams, ActivatorParams 
from QGCN.activator import QGCNModel, QGCNDataSet

# sets the parameters of the dataset:
external = ExternalParams(file_path="./data/Mutagenicity_external_data_all.csv",
                          graph_col="g_id", node_col="node",
                          embeddings=["chem"], continuous=[])
graphs_data = GraphsDataParams(file_path="../src/QGCN/data/Mutagenicity_all.csv",
                               standardization="min_max")

# sets the parameters of the model:
model = ModelParams(label_type="binary", num_classes=2, use_embeddings="True", embeddings_dim=[10],
                    activation="srss_", dropout=0.2, lr=0.005, optimizer="ADAM_", L2_regularization=0.005, f="x1_x0",
                    GCN_layers=[
                        {"in_dim": "None", "out_dim": 250},
                        {"in_dim": 250, "out_dim": 100}])
activator = ActivatorParams(epochs=100)

qgcn_model = QGCNModel("Mutagen", graphs_data, external, model, activator)
qgcn_model.train(should_print=False)

ds = QGCNDataSet("Mutagen", graphs_data, external)
loader = DataLoader(
    ds.get_dataset(),
    shuffle=False
)

for _, (A, x0, embed, label) in enumerate(loader):
    output = qgcn_model.predict(A, x0, embed)
    print(output, label)
```

## Links
The datasets can be download here: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets . Notice you will have to change their format to ours. You can see an example data here (gitHub link) the conventor in datasets -> change_data_format.py
Mail address for more information: 123shovalf@gmail.com
