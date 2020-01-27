import time

from torch.nn.functional import relu, softmax, cross_entropy
from torch.optim import Adam
from params.parameters import BilinearDatasetParams, BilinearActivatorParams, BilinearLayerParams, LinearLayerParams, \
    LayeredBilinearModuleParams, DEG, CENTRALITY, BFS, NORM_REDUCED, ExternalDataParams


# ---------------------------------------------------  PROTEIN ---------------------------------------------------------
class ProteinTrainExternalDataParams(ExternalDataParams):
    def __init__(self):
        super().__init__()
        self.GRAPH_COL = "g_id"
        self.NODE_COL = "node"
        self.FILE_NAME = "Protein_external_data_train.csv"
        self.EMBED_COLS = ["type"]
        self.VALUE_COLS = ["aaLength"]


class ProteinDevExternalDataParams(ProteinTrainExternalDataParams):
    def __init__(self):
        super().__init__()
        self.FILE_NAME = "Protein_external_data_dev.csv"


class ProteinTestExternalDataParams(ProteinTrainExternalDataParams):
    def __init__(self):
        super().__init__()
        self.FILE_NAME = "Protein_external_data_test.csv"


class ProteinAllExternalDataParams(ProteinTrainExternalDataParams):
    def __init__(self):
        super().__init__()
        self.FILE_NAME = "Protein_external_data_all.csv"


class ProteinDatasetTrainParams(BilinearDatasetParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Protein_train"
        self.DATASET_FILENAME = "Protein_train.csv"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.GRAPH_NAME_COL = "g_id"
        self.LABEL_COL = "label"
        self.PERCENTAGE = 1
        self.DIRECTED = False
        self.FEATURES = [DEG, CENTRALITY, BFS]


class ProteinDatasetDevParams(ProteinDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Protein_dev"
        self.DATASET_FILENAME = "Protein_dev.csv"


class ProteinDatasetTestParams(ProteinDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Protein_test"
        self.DATASET_FILENAME = "Protein_test.csv"


class ProteinDatasetAllParams(ProteinDatasetTrainParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "Protein_all"
        self.DATASET_FILENAME = "Protein_all.csv"
# ----------------------------------------------------------------------------------------------------------------------


class ProteinBilinearLayerParams(BilinearLayerParams):
    def __init__(self, in_col_dim, ftr_len):
        super().__init__(in_col_dim, ftr_len)
        self.LEFT_LINEAR_ROW_DIM = in_col_dim   # should be equal to RIGHT_LINEAR_IN and FirstLayerModelParams::OUT_DIM
        self.LEFT_LINEAR_COL_DIM = 1            # out rows
        self.RIGHT_LINEAR_ROW_DIM = ftr_len     # should be equal to FirstLayerModelParams::ROW_DIM
        self.RIGHT_LINEAR_COL_DIM = 6           # out cols
        self.ACTIVATION_FUNC = lambda x: x
        self.ACTIVATION_FUNC_ARGS = {}


class ProteinLinearLayerParams(LinearLayerParams):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__(in_dim, out_dim, dropout)
        self.ROW_DIM = in_dim
        self.COL_DIM = out_dim
        self.ACTIVATION_FUNC = relu
        self.DROPOUT = dropout


class ProteinLayeredBilinearModuleParams(LayeredBilinearModuleParams):
    def __init__(self, ftr_len=6, layer_dim=None, embed_vocab_dim=None):
        super().__init__(ftr_len, layer_dim, embed_vocab_dim)
        self.IS_BINARY = False
        self.EMBED_DIMS = [10]
        self.NORM = NORM_REDUCED
        self.DROPOUT = 0.5
        self.LR = 1e-4
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 1e-3

        if layer_dim is None:
            self.NUM_LAYERS = 3
            self.LINEAR_PARAMS_LIST = [
                ProteinLinearLayerParams(in_dim=ftr_len, out_dim=500, dropout=self.DROPOUT),
                ProteinLinearLayerParams(in_dim=500, out_dim=250, dropout=self.DROPOUT),
                ProteinLinearLayerParams(in_dim=250, out_dim=125, dropout=self.DROPOUT),
                ProteinLinearLayerParams(in_dim=200, out_dim=1, dropout=self.DROPOUT)
            ]
        self.BILINEAR_PARAMS = ProteinBilinearLayerParams(self.LINEAR_PARAMS_LIST[self.NUM_LAYERS - 1].COL_DIM,
                                                          self.LINEAR_PARAMS_LIST[0].ROW_DIM)


class ProteinBilinearActivatorParams(BilinearActivatorParams):
    def __init__(self):
        super().__init__()
        self.DEV_SPLIT = 0.333
        self.TEST_SPLIT = 0.333
        self.LOSS = cross_entropy  # f.factor_loss  #
        self.BATCH_SIZE = 32
        self.EPOCHS = 300
        self.DATASET = "Protein - MultiClass"


if __name__ == '__main__':
    from bilinear_model import LayeredBilinearModule
    from dataset.dataset_model import BilinearDataset
    from dataset.dataset_external_data import ExternalData
    from multi_class_bilinear_activator import BilinearMultiClassActivator

    t = time.time()

    ALL = True
    if ALL == True:
        ext_train = ExternalData(ProteinAllExternalDataParams())
        aids_train_ds = BilinearDataset(ProteinDatasetAllParams(), external_data=ext_train)

        activator = BilinearMultiClassActivator(LayeredBilinearModule(ProteinLayeredBilinearModuleParams(
            ftr_len=aids_train_ds.len_features, embed_vocab_dim=ext_train.len_embed())),
                                      ProteinBilinearActivatorParams(), aids_train_ds)
        activator.train()

    if ALL == False:
        ext_train = ExternalData(ProteinTrainExternalDataParams())
        ext_dev = ExternalData(ProteinDevExternalDataParams(), idx_to_symbol=ext_train.idx_to_symbol_dict)
        ext_test = ExternalData(ProteinTestExternalDataParams(), idx_to_symbol=ext_train.idx_to_symbol_dict)

        protein_train_ds = BilinearDataset(ProteinDatasetTrainParams(), external_data=ext_train)
        protein_dev_ds = BilinearDataset(ProteinDatasetDevParams(), external_data=ext_dev)
        protein_test_ds = BilinearDataset(ProteinDatasetTestParams(), external_data=ext_test)

        activator = BilinearMultiClassActivator(LayeredBilinearModule(
            ProteinLayeredBilinearModuleParams(ftr_len=protein_train_ds.len_features)), ProteinBilinearActivatorParams(),
            protein_train_ds, dev_data=protein_dev_ds, test_data=protein_test_ds)
        activator.train()

    print("total time", time.time() - t)
