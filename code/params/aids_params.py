from torch import sigmoid
from torch.nn.functional import relu
from torch.optim import Adam
from torch.nn import functional
import os
from betweenness_centrality import BetweennessCentralityCalculator
from bfs_moments import BfsMomentsCalculator
from bilinear_activator import BilinearActivator
from bilinear_model import LayeredBilinearModule
from dataset.dataset_model import BilinearDataset
from dataset.dataset_external_data import ExternalData
from feature_calculators import FeatureMeta
from params.parameters import BilinearDatasetParams, LayeredBilinearModuleParams, LinearLayerParams, \
    BilinearLayerParams, BilinearActivatorParams, DEG, CENTRALITY, BFS, NORM_REDUCED, ExternalDataParams


# ------------------------------------------------------  AIDS ---------------------------------------------------------


class AidsTrainExternalDataParams(ExternalDataParams):
    def __init__(self):
        super().__init__()
        self.GRAPH_COL = "g_id"
        self.NODE_COL = "node"
        self.FILE_NAME = "AIDS_external_data_train.csv"
        self.EMBED_COLS = ["chem", "symbol"]
        self.VALUE_COLS = ["charge", "x", "y"]


class AidsDevExternalDataParams(AidsTrainExternalDataParams):
    def __init__(self):
        super().__init__()
        self.FILE_NAME = "AIDS_external_data_dev.csv"


class AidsTestExternalDataParams(AidsTrainExternalDataParams):
    def __init__(self):
        super().__init__()
        self.FILE_NAME = "AIDS_external_data_test.csv"


class AidsAllExternalDataParams(AidsTrainExternalDataParams):
    def __init__(self):
        super().__init__()
        self.FILE_NAME = "AIDS_external_data_all.csv"


class AidsDatasetTrainParams(BilinearDatasetParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "AIDS_train"
        self.DATASET_FILENAME = "AIDS_train.csv"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.GRAPH_NAME_COL = "g_id"
        self.LABEL_COL = "label"
        self.PERCENTAGE = 1
        self.DIRECTED = False
        self.FEATURES = [CENTRALITY, BFS]


class AidsDatasetDevParams(AidsDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "AIDS_dev"
        self.DATASET_FILENAME = "AIDS_dev.csv"


class AidsDatasetTestParams(AidsDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "AIDS_test"
        self.DATASET_FILENAME = "AIDS_test.csv"


class AidsDatasetAllParams(AidsDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "AIDS_all"
        self.DATASET_FILENAME = "AIDS_all.csv"
# ----------------------------------------------------------------------------------------------------------------------


class AidsBilinearLayerParams(BilinearLayerParams):
    def __init__(self, in_col_dim, ftr_len):
        super().__init__(in_col_dim, ftr_len)
        self.LEFT_LINEAR_ROW_DIM = in_col_dim   # should be equal to RIGHT_LINEAR_IN and FirstLayerModelParams::OUT_DIM
        self.LEFT_LINEAR_COL_DIM = 1            # out rows
        self.RIGHT_LINEAR_ROW_DIM = ftr_len     # should be equal to FirstLayerModelParams::ROW_DIM
        self.RIGHT_LINEAR_COL_DIM = 1           # out cols
        self.ACTIVATION_FUNC = sigmoid


class AidsLinearLayerParams(LinearLayerParams):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__(in_dim, out_dim, dropout)
        self.NORM = NORM_REDUCED
        self.ROW_DIM = in_dim
        self.COL_DIM = out_dim
        self.ACTIVATION_FUNC = relu
        self.DROPOUT = dropout


class AidsLayeredBilinearModuleParams(LayeredBilinearModuleParams):
    def __init__(self, ftr_len=6, layer_dim=None, embed_vocab_dim=None):
        super().__init__(ftr_len, layer_dim, embed_vocab_dim)
        self.EMBED_DIMS = [20, 20]
        self.DROPOUT = 0
        self.LR = 1e-3
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 0

        if layer_dim is None:
            self.NUM_LAYERS = 2
            self.LINEAR_PARAMS_LIST = [
                AidsLinearLayerParams(in_dim=ftr_len, out_dim=100, dropout=self.DROPOUT),
                AidsLinearLayerParams(in_dim=100, out_dim=50, dropout=self.DROPOUT),
                AidsLinearLayerParams(in_dim=100, out_dim=50, dropout=self.DROPOUT)
            ]
        self.BILINEAR_PARAMS = AidsBilinearLayerParams(self.LINEAR_PARAMS_LIST[self.NUM_LAYERS - 1].COL_DIM,
                                                       self.LINEAR_PARAMS_LIST[0].ROW_DIM)


class AidsBilinearActivatorParams(BilinearActivatorParams):
    def __init__(self):
        super().__init__()
        self.DEV_SPLIT = 0.1153
        self.TEST_SPLIT = 0.538
        self.LOSS = functional.binary_cross_entropy_with_logits  # f.factor_loss  #
        self.BATCH_SIZE = 128
        self.EPOCHS = 250
        self.DATASET = "Aids"


if __name__ == '__main__':
    ALL = True
    if ALL == True:
        ext_train = ExternalData(AidsAllExternalDataParams())
        aids_train_ds = BilinearDataset(AidsDatasetAllParams(), external_data=ext_train)

        activator = BilinearActivator(LayeredBilinearModule(AidsLayeredBilinearModuleParams(
            ftr_len=aids_train_ds.len_features, embed_vocab_dim=ext_train.len_embed())),
                                      AidsBilinearActivatorParams(), aids_train_ds)
        activator.train()

    if ALL == False:
        ext_train = ExternalData(AidsTrainExternalDataParams())
        ext_dev = ExternalData(AidsDevExternalDataParams(), idx_to_symbol=ext_train.idx_to_symbol_dict)
        ext_test = ExternalData(AidsTestExternalDataParams(), idx_to_symbol=ext_train.idx_to_symbol_dict)

        aids_train_ds = BilinearDataset(AidsDatasetTrainParams(), external_data=ext_train)
        aids_dev_ds = BilinearDataset(AidsDatasetDevParams(), external_data=ext_dev)
        aids_test_ds = BilinearDataset(AidsDatasetTestParams(), external_data=ext_test)

        activator = BilinearActivator(LayeredBilinearModule(
            AidsLayeredBilinearModuleParams(ftr_len=aids_train_ds.len_features, embed_vocab_dim=ext_train.len_embed())),
                                      AidsBilinearActivatorParams(), aids_train_ds, dev_data=aids_dev_ds,
                                      test_data=aids_test_ds)
        activator.train()
