import sys
import os
f = open("curr_pwd", "wt")
cwd = os.getcwd()
f.write(cwd)
f.close()

sys.path.insert(1, os.path.join(cwd, ".."))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graph-measures"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graph-measures", "features_algorithms"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graph-measures", "graph_infra"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graph-measures", "features_infra"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graph-measures", "features_meta"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graph-measures", "features_algorithms", "vertices"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graphs-package", "features_processor"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graphs-package", "multi_graph"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graphs-package", "temporal_graphs"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graphs-package", "features_processor", "motif_variations"))
sys.path.insert(1, os.path.join(cwd, "..", "..", "..", "graphs-package"))

from bilinear_activator import BilinearActivator
from bilinear_model import LayeredBilinearModule
from dataset.dataset_external_data import ExternalData
from dataset.dataset_model import BilinearDataset
from multi_class_bilinear_activator import BilinearMultiClassActivator
import nni
import logging
from params.protein_params import ProteinDatasetAllParams, ProteinLayeredBilinearModuleParams, \
    ProteinBilinearActivatorParams, ProteinAllExternalDataParams
from params.aids_params import AidsDatasetAllParams, AidsAllExternalDataParams, AidsLayeredBilinearModuleParams, \
    AidsBilinearActivatorParams
from params.grec_params import GrecDatasetAllParams, GrecLayeredBilinearModuleParams, GrecBilinearActivatorParams, \
    GrecAllExternalDataParams
from params.mutagen_params import MutagenDatasetAllParams, MutagenLayeredBilinearModuleParams, \
    MutagenBilinearActivatorParams, MutagenAllExternalDataParams
from params.parameters import BFS, CENTRALITY, DEG


logger = logging.getLogger("NNI_logger")
from torch.optim import Adam, SGD
ADAM = Adam
SGD = SGD
NONE = None


def run_trial(params, dataset_param_class, module_param_class, activator_param_class, ext_data, is_multi_class):
    ds_params = dataset_param_class()
    ds_params.FEATURES = [globals()[ftr] for ftr in params['input_vec']]
    dataset = BilinearDataset(ds_params, external_data=ext_data)

    # model
    layers = []
    for dim1, dim2 in zip([0] + params['layers_config'], params['layers_config']):
        layers.append((int(dim1), int(dim2)))
    # if params['layers_config']["_name"] == "2_layers":
        # layers.append([None, int(params['layers_config']["h1_dim"])])
        # layers.append([int(params['layers_config']["h1_dim"]), int(params['layers_config']["h2_dim"])])

    # elif params['layers_config']["_name"] == "3_layers":
        # layers.append([None, int(params['layers_config']["h1_dim"])])
        # layers.append([int(params['layers_config']["h1_dim"]), int(params['layers_config']["h2_dim"])])
        # layers.append([int(params['layers_config']["h2_dim"]), int(params['layers_config']["h3_dim"])])

    model_params = module_param_class(ftr_len=dataset.len_features, layer_dim=layers,
                                      embed_vocab_dim=ext_data.len_embed())
    model_params.DROPOUT = params['dropout']
    model_params.WEIGHT_DECAY = params['regularization']
    model_params.LR = params['learning_rate']
    model_params.OPTIMIZER = globals()[params['optimizer']]

    # activator
    activator_params = activator_param_class()
    activator_params.BATCH_SIZE = params['batch_size']
    activator_params.EPOCHS = params['epochs']

    model = LayeredBilinearModule(model_params)
    activator = BilinearMultiClassActivator(model, activator_params, dataset, nni=True) if is_multi_class else \
        BilinearActivator(model, activator_params, dataset, nni=True)
    activator.train(show_plot=False, early_stop=True)


def main(data):
    try:
        # get parameters form tuner
        params = nni.get_next_parameter()
        logger.debug(params)
        run_trial(params, *get_params_by_dataset(data))
    except Exception as exception:
        logger.error(exception)
        raise


def get_params_by_dataset(data):
    dict_classes = {
        "AIDS": [AidsDatasetAllParams, AidsLayeredBilinearModuleParams, AidsBilinearActivatorParams,
                 ExternalData(AidsAllExternalDataParams()), False],
        "PROTEIN": [ProteinDatasetAllParams, ProteinLayeredBilinearModuleParams, ProteinBilinearActivatorParams,
                    ExternalData(ProteinAllExternalDataParams()), True],
        "MUTAGEN": [MutagenDatasetAllParams, MutagenLayeredBilinearModuleParams, MutagenBilinearActivatorParams,
                    ExternalData(MutagenAllExternalDataParams()), False],
        "GREC": [GrecDatasetAllParams, GrecLayeredBilinearModuleParams, GrecBilinearActivatorParams,
                    ExternalData(GrecAllExternalDataParams()), True]
    }
    return dict_classes[data]


if __name__ == "__main__":
    data = sys.argv[1]
    main(data)
