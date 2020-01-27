from torch import tanh
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
from params.parameters import BilinearLayerParams, LinearLayerParams, LayeredBilinearModuleParams, \
    BilinearActivatorParams, BilinearDatasetParams, DEG, CENTRALITY, BFS, NORM_REDUCED, ExternalDataParams


# ---------------------------------------------------  MUTAGEN ---------------------------------------------------------
class MutagenTrainExternalDataParams(ExternalDataParams):
    def __init__(self):
        super().__init__()
        self.GRAPH_COL = "g_id"
        self.NODE_COL = "node"
        self.FILE_NAME = "Mutagenicity_external_data_train.csv"
        self.EMBED_COLS = ["chem"]
        self.VALUE_COLS = []


class MutagenDevExternalDataParams(MutagenTrainExternalDataParams):
    def __init__(self):
        super().__init__()
        self.FILE_NAME = "Mutagenicity_external_data_dev.csv"


class MutagenTestExternalDataParams(MutagenTrainExternalDataParams):
    def __init__(self):
        super().__init__()
        self.FILE_NAME = "Mutagenicity_external_data_test.csv"


class MutagenAllExternalDataParams(MutagenTrainExternalDataParams):
    def __init__(self):
        super().__init__()
        self.FILE_NAME = "Mutagenicity_external_data_all.csv"


class MutagenDatasetTrainParams(BilinearDatasetParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Mutagenicity_train"
        self.DATASET_FILENAME = "Mutagenicity_train.csv"
        self.EXTERNAL_DATA_FILNAME = "Mutagenicity_external_data_train.csv"
        self.EMBED_COL = []
        self.VAL_COL = []
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.GRAPH_NAME_COL = "g_id"
        self.LABEL_COL = "label"
        self.PERCENTAGE = 1
        self.DIRECTED = False
        self.FEATURES = [DEG, BFS, CENTRALITY]  # , CENTRALITY, BFS]


class MutagenDatasetDevParams(MutagenDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Mutagenicity_dev"
        self.DATASET_FILENAME = "Mutagenicity_dev.csv"


class MutagenDatasetTestParams(MutagenDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Mutagenicity_test"
        self.DATASET_FILENAME = "Mutagenicity_test.csv"


class MutagenDatasetAllParams(MutagenDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Mutagenicity_all"
        self.DATASET_FILENAME = "Mutagenicity_all.csv"

# ----------------------------------------------------------------------------------------------------------------------


class MutagenBilinearLayerParams(BilinearLayerParams):
    def __init__(self, in_col_dim, ftr_len):
        super().__init__(in_col_dim, ftr_len)
        self.LEFT_LINEAR_ROW_DIM = in_col_dim   # should be equal to RIGHT_LINEAR_IN and FirstLayerModelParams::OUT_DIM
        self.LEFT_LINEAR_COL_DIM = 1            # out rows
        self.RIGHT_LINEAR_ROW_DIM = ftr_len     # should be equal to FirstLayerModelParams::ROW_DIM
        self.RIGHT_LINEAR_COL_DIM = 1           # out cols
        self.ACTIVATION_FUNC = relu


class MutagenLinearLayerParams(LinearLayerParams):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__(in_dim, out_dim, dropout)
        self.ROW_DIM = in_dim
        self.COL_DIM = out_dim
        self.ACTIVATION_FUNC = relu
        self.DROPOUT = dropout


class MutagenLayeredBilinearModuleParams(LayeredBilinearModuleParams):
    def __init__(self, ftr_len=6, layer_dim=None, embed_vocab_dim=None):
        super().__init__(ftr_len, layer_dim, embed_vocab_dim)
        self.EMBED_DIMS = [10]
        self.NORM = NORM_REDUCED
        self.DROPOUT = 0
        self.LR = 1e-3
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 0.8

        if layer_dim is None:
            self.NUM_LAYERS = 2
            self.LINEAR_PARAMS_LIST = [
                MutagenLinearLayerParams(in_dim=ftr_len, out_dim=50, dropout=self.DROPOUT),
                MutagenLinearLayerParams(in_dim=50, out_dim=25, dropout=self.DROPOUT),
                MutagenLinearLayerParams(in_dim=50, out_dim=10, dropout=self.DROPOUT),
                MutagenLinearLayerParams(in_dim=200, out_dim=1, dropout=self.DROPOUT)
            ]
        self.BILINEAR_PARAMS = MutagenBilinearLayerParams(self.LINEAR_PARAMS_LIST[self.NUM_LAYERS - 1].COL_DIM,
                                                          self.LINEAR_PARAMS_LIST[0].ROW_DIM)


class MutagenBilinearActivatorParams(BilinearActivatorParams):
    def __init__(self):
        super().__init__()
        self.DEV_SPLIT = 0.125
        self.TEST_SPLIT = 0.75
        self.LOSS = functional.binary_cross_entropy_with_logits  # f.factor_loss  #
        self.BATCH_SIZE = 8
        self.EPOCHS = 200
        self.DATASET = "Mutagen"


if __name__ == '__main__':
    ALL = True
    if ALL == True:
        ext_train = ExternalData(MutagenAllExternalDataParams())
        aids_train_ds = BilinearDataset(MutagenDatasetAllParams(), external_data=ext_train)

        activator = BilinearActivator(LayeredBilinearModule(MutagenLayeredBilinearModuleParams(
            ftr_len=aids_train_ds.len_features, embed_vocab_dim=ext_train.len_embed())),
                                      MutagenBilinearActivatorParams(), aids_train_ds)
        activator.train()
    if ALL == False:
        ext_train = ExternalData(MutagenTrainExternalDataParams())
        ext_dev = ExternalData(MutagenDevExternalDataParams(), idx_to_symbol=ext_train.idx_to_symbol_dict)
        ext_test = ExternalData(MutagenTestExternalDataParams(), idx_to_symbol=ext_train.idx_to_symbol_dict)

        mutagen_train_ds = BilinearDataset(MutagenDatasetTrainParams(), external_data=ext_train)
        mutagen_dev_ds = BilinearDataset(MutagenDatasetDevParams(), external_data=ext_dev)
        mutagen_test_ds = BilinearDataset(MutagenDatasetTestParams(), external_data=ext_test)

        activator = BilinearActivator(LayeredBilinearModule(MutagenLayeredBilinearModuleParams(ftr_len=mutagen_train_ds.len_features, embed_vocab_dim=ext_train.len_embed())),
                                      MutagenBilinearActivatorParams(), mutagen_train_ds, dev_data=mutagen_dev_ds, test_data=mutagen_test_ds)
        activator.train()
