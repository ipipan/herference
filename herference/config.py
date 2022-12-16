from pathlib import Path
from dataclasses import dataclass, field
import yaml

from transformers import (
    RobertaConfig,
)
from torch._C import device


def get_config():
    with open(Path(__file__).parent / 'package_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)


@dataclass
class ModelConfig:
    max_span_length: int = 30
    max_model_input_length: int = 510
    top_lambda: float = 0.5
    ffnn_size: int = 3072
    normalise_loss: bool = True
    dropout_prob: float = 0.5
    seed: int = 98765432
    hidden_size = 768
    layer_norm_eps = 1e-5
    base_model_prefix = 'roberta'
    max_total_seq_len = 4000
    null_id_for_coref = 0

    def to_transformers_config(self):
        return RobertaConfig(
            max_position_embeddings=self.max_span_length,
            hidden_dropout_prob=self.dropout_prob,
            attention_probs_dropout_prob=self.dropout_prob,
            seed=self.seed,
            hidden_size=self.hidden_size,
            layer_norm_eps=self.layer_norm_eps,
        )


@dataclass
class Config:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    LOG_LEVEL = 'INFO'
    MODEL_CONFIG_CLASS = RobertaConfig
    MODEL_NAME = 'ipipan/herbert-large-coref'
    DEVICE: device = device('cpu')

    def __post_init__(self):
        self.model_config = ModelConfig(**self.model_config)

