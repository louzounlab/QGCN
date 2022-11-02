from dataclasses import asdict

from torch.utils.data import DataLoader

from QGCN.params import GraphsDataParams, ExternalParams, ModelParams, ActivatorParams
from QGCN.dataset.dataset_graphs_model import GraphsDataset
from QGCN.dataset.dataset_external_data import ExternalData
from QGCN.QGCN_model.qgcn_activator import QGCNActivator
from QGCN.QGCN_model.QGCN import QGCN


class QGCNDataSet:
    """
    This class gets the parameters of the data (as external_params and graph_data_params objects),
    and build from them datasets.
    """
    def __init__(self, dataset_name: str,
                 graph_data_params: GraphsDataParams = None,
                 external_params: ExternalParams = None,
                 params_file: str = None):

        if params_file:
            self.params = params_file
        elif graph_data_params and external_params:
            self.params = {
                "dataset_name": dataset_name,
                "external": asdict(external_params),
                "graphs_data": asdict(graph_data_params)
            }
        else:
            raise Exception('QGCNDataset should gets a params file or 2 params objects.')

        ext_train = ExternalData(self.params)
        self.dataset = GraphsDataset(self.params, external_data=ext_train)

    def get_dataset(self):
        return self.dataset


class QGCNModel:
    """
    This class create a dataset and a pytorch model of QGCN.
    You should init the model with the params dataclasses, and then call the train method.
    """
    def __init__(self, dataset_name: str,
                 graph_data_params: GraphsDataParams = None,
                 external_params: ExternalParams = None,
                 model_params: ModelParams = None,
                 activator_params: ActivatorParams = None,
                 device="cpu", params_file: str = None):

        if params_file:
            self.params = params_file
        elif graph_data_params and external_params and model_params and activator_params:
            self.params = {
                "dataset_name": dataset_name,
                "external": asdict(external_params),
                "graphs_data": asdict(graph_data_params),
                "model": asdict(model_params),
                "activator": asdict(activator_params)
            }
        else:
            raise Exception('QGCNModel should gets a params-file-path or 4 params objects.')

        self.ext_train = ExternalData(self.params)
        self.ds = GraphsDataset(self.params, external_data=self.ext_train)
        qgcn = QGCN(self.params, self.ds.len_features, self.ext_train.len_embed())
        self.activator = QGCNActivator(qgcn, self.params, self.ds, device=device)

    def get_dataset(self):
        return self.ds

    def train(self, should_print=True):
        self.activator.train(should_print=should_print)

    def predictl(self, dataloader: DataLoader):
        outputs = []
        model = self.activator.model
        model.eval()
        for _, (A, x0, embed) in enumerate(dataloader):
            outputs.append(model(A, x0, embed))

        return outputs

    def predict(self, A, x0, embed):
        model = self.activator.model
        model.eval()
        return model(A, x0, embed)

    def get_model(self):
        return self.activator.model
