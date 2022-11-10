from QGCN.activator import QGCNModel, QGCNDataSet
from torch.utils.data import DataLoader

# qgcn_model = QGCNModel(dataset_name="Mutagen", params_file="./params/mutagen_params.json")
qgcn_model = QGCNModel(dataset_name="Aids", params_file="./params/aids_params.json")
qgcn_model.train()

# ds = QGCNDataSet("Aids", params_file="./params/default_binary_params.json")
# loader = DataLoader(
#     ds.get_dataset(),
#     shuffle=False
# )

# for _, (A, x0, embed, label) in enumerate(loader):
#     output = qgcn_model.predict(A, x0, embed)
#     print(output, label)