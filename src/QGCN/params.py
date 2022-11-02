from dataclasses import dataclass, field


def get_embeddings_dim():
    return [10, 10]


def get_GCN_layers():
    return [
        {"in_dim": "None", "out_dim": 100},
        {"in_dim": 100, "out_dim": 50},
        {"in_dim": 50, "out_dim": 25}
    ]


def get_default_features():
    return ["DEG", "CENTRALITY", "BFS"]


def get_embeddings():
    return ["chem", "symbol"]


def get_continuous():
    return ["charge", "x", "y"]


@dataclass
class ModelParams:
    label_type: str = "binary"
    num_classes: int = 2
    use_embeddings: str = "True"
    embeddings_dim: list[int] = field(default_factory=get_embeddings_dim)
    activation: str = "relu_"
    dropout: float = 0
    lr: float = 1e-3
    optimizer: str = "ADAM_"
    L2_regularization: float = 0
    f: str = "c_x0"
    GCN_layers: list[dict] = field(default_factory=get_GCN_layers)


@dataclass
class ActivatorParams:
    epochs: int = 3
    batch_size: int = 128
    loss_func: str = "binary_cross_entropy_with_logits_"
    train: float = 0.3467
    dev: float = 0.1153
    test: float = 0.538


@dataclass
class GraphsDataParams:
    file_path: str = "../../test/data/AIDS_all.csv"
    graph_col: str = "g_id"
    src_col: str = "src"
    dst_col: str = "dst"
    label_col: str = "label"
    directed: str = "False"
    features: list[str] = field(default_factory=get_default_features)
    adjacency_norm: str = "NORM_REDUCED"
    percentage: float = 1
    standardization: str = "zscore"


@dataclass
class ExternalParams:
    file_path: str = "../../test/data/AIDS_external_data_all.csv"
    graph_col: str = "g_id"
    node_col: str = "node"
    embeddings: list[str] = field(default_factory=get_embeddings)
    continuous: list[str] = field(default_factory=get_continuous)
