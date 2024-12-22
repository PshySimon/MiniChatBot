from transformers import PretrainedConfig


class MiniChatConfig(PretrainedConfig):
    model_type = "mini_chat"

    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 16,
        n_kv_heads: int = 8,
        vocab_size: int = 6400,
        hidden_dim: int = None,
        multiple_of: int = 64,
        norm_eps: float = 1e-5,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        flash_attn: bool = True,
        **kwargs,
    ):
        # 模型的维度
        self.dim = dim
        # 模型transformer层数
        self.n_layers = n_layers
        # 多头注意力，query矩阵头的数量
        self.n_heads = n_heads
        # 多头注意力，key和value矩阵头的数量
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)