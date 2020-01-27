from torch.nn import Module, Linear, Dropout, Embedding, ModuleList
import torch
from torch.utils.data import DataLoader

from params.parameters import LinearLayerParams, BilinearLayerParams, NORM_REDUCED, \
    NORM_REDUCED_SYMMETRIC, LayeredBilinearModuleParams

"""
given A, x0 : A=Adjacency_matrix, x0=nodes_vec
First_model => x1(n x k) = sigma( A(n x n) * x0(n x d) * W1(d x k) )
Bilinear_model => x2(1 x 1) = sigma( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
"""


class LinearLayer(Module):
    def __init__(self, params: LinearLayerParams):
        super(LinearLayer, self).__init__()
        # useful info in forward function
        self._linear = Linear(params.ROW_DIM, params.COL_DIM)
        self._activation = params.ACTIVATION_FUNC
        self._dropout = Dropout(p=params.DROPOUT) if params.DROPOUT else None
        self._gpu = False

    def gpu_device(self, is_gpu: bool):
        self._gpu = is_gpu

    def _sync(self):
        if self._gpu:
            torch.cuda.synchronize()

    def forward(self, A, x0):
        # Dropout layer
        x0 = self._dropout(x0) if self._dropout else x0
        # tanh( A(n x n) * x0(n x d) * W1(d x k) )
        Ax = torch.matmul(A, x0)
        self._sync()

        x = self._linear(Ax)
        x1 = self._activation(x)
        return x1


class BilinearLayer(Module):
    def __init__(self, params: BilinearLayerParams):
        super(BilinearLayer, self).__init__()
        # useful info in forward function
        self._left_linear = Linear(params.LEFT_LINEAR_ROW_DIM, params.LEFT_LINEAR_COL_DIM)
        self._right_linear = Linear(params.RIGHT_LINEAR_ROW_DIM, params.RIGHT_LINEAR_COL_DIM)
        self._activation = params.ACTIVATION_FUNC
        self._activation_args = params.ACTIVATION_FUNC_ARGS
        self._gpu = False

    def gpu_device(self, is_gpu: bool):
        self._gpu = is_gpu

    def _sync(self):
        if self._gpu:
            torch.cuda.synchronize()

    def forward(self, A, x0, x1):
        # sigmoid( W2(1 x k) * trans(x1)(k x n) * A(n x n) * x0(n x d) * W1(d x 1) )
        x1_A = torch.matmul(x1.permute(0, 2, 1), A)
        self._sync()
        W2_x1_A = self._left_linear(x1_A.permute(0, 2, 1))
        W2_x1_A_x0 = torch.matmul(W2_x1_A.permute(0, 2, 1), x0)
        self._sync()
        W2_x1_A_x0_W3 = self._right_linear(W2_x1_A_x0)
        x2 = self._activation(W2_x1_A_x0_W3, **self._activation_args)
        return x2


class LayeredBilinearModule(Module):
    """
    first linear layer is executed numerous times
    """

    def __init__(self, params: LayeredBilinearModuleParams):
        super(LayeredBilinearModule, self).__init__()

        # add embedding layers
        self._is_binary = params.IS_BINARY
        self._is_embed = False if params.EMBED_VOCAB_DIMS is None else True
        if self._is_embed:
            # embeddings are added to ftr vector -> update dimensions of relevant weights
            params.LINEAR_PARAMS_LIST[0].ROW_DIM += sum(params.EMBED_DIMS)
            params.BILINEAR_PARAMS.RIGHT_LINEAR_ROW_DIM += sum(params.EMBED_DIMS)

            # add embedding layers
            self._embed_layers = []
            for vocab_dim, embed_dim in zip(params.EMBED_VOCAB_DIMS, params.EMBED_DIMS):
                self._embed_layers.append(Embedding(vocab_dim, embed_dim))
            self._embed_layers = ModuleList(self._embed_layers)

        self._num_layers = params.NUM_LAYERS
        self._linear_layers = []
        # create linear layers
        for i in range(params.NUM_LAYERS):
            self._linear_layers.append(LinearLayer(params.LINEAR_PARAMS_LIST[i]))
        self._linear_layers = ModuleList(self._linear_layers)

        self._bilinear_layer = BilinearLayer(params.BILINEAR_PARAMS)
        self.optimizer = self.set_optimizer(params.LR, params.OPTIMIZER, params.WEIGHT_DECAY)

        self._gpu = False

    def gpu_device(self, is_gpu: bool):
        self._gpu = is_gpu
        for layer in self._linear_layers:
            layer.gpu_device(is_gpu)
        self._bilinear_layer.gpu_device(is_gpu)

    def _sync(self):
        if self._gpu:
            torch.cuda.synchronize()

    def set_optimizer(self, lr, opt, weight_decay):
        return opt(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _fix_shape(self, input):
        return input if len(input.shape) == 3 else input.unsqueeze(dim=0)

    def forward(self, A, x0, embed):
        if self._is_embed:
            list_embed = []
            for i, embedding in enumerate(self._embed_layers):
                list_embed.append(embedding(embed[:, :, i]))
            x0 = torch.cat([x0] + list_embed, dim=2)

        x1 = x0
        self._sync()
        for i in range(self._num_layers):
            x1 = self._linear_layers[i](A, x1)
        x2 = self._bilinear_layer(A, x0, x1)
        # x2 = torch.sigmoid(x2) if self._is_binary else torch.softmax(x2, dim=1)
        return x2


if __name__ == "__main__":
    from dataset.datset_sampler import ImbalancedDatasetSampler
    from params.aids_params import AidsAllExternalDataParams, AidsDatasetAllParams
    from dataset.dataset_external_data import ExternalData
    from dataset.dataset_model import BilinearDataset

    ext_train = ExternalData(AidsAllExternalDataParams())
    ds = BilinearDataset(AidsDatasetAllParams(), external_data=ext_train)
    dl = DataLoader(
        dataset=ds,
        collate_fn=ds.collate_fn,
        batch_size=64,
        sampler=ImbalancedDatasetSampler(ds)
    )
    m_params = LayeredBilinearModuleParams(ftr_len=ds.len_features, embed_vocab_dim=ext_train.len_embed())
    m_params.EMBED_DIMS = [20, 20]
    module = LayeredBilinearModule(m_params)
    # module = BilinearModule(BilinearModuleParams())
    for i, (_A, _D, _x0, _l) in enumerate(dl):
        _x2 = module(_A, _D, _x0)
        e = 0
