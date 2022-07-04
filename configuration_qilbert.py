""" QIL-BERT configuration """
from collections import OrderedDict
from typing import Mapping

from transformers.onnx import OnnxConfig
import sys
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class QILBertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a ['QILBertModel']. 
    It is used to instantiate a QIL-BERT model according to the specified arugments,

    Configuration objects inherit from ['PretrainedConfig'] and can be used to control the model outputs.
    Read the documentation from ['PretrainedConfig'] for more information.

    Args:
        vocab_size('int', *optional*, defaults to 30522):
            Vocabulary size of the QIL-BERT model. Defines the number of different tokens that can be represented
            by the 'input_ids', passed when calling ['QILBertModel']
        hidden_size('int', *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers ('int', *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads ('int', *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size ('int', *optional*, defaults to 3072):
            Dimensionality of the "intermediate" ( often named feed-forward ) layer in the Transformer encoder.
        hidden_act ('str' or 'Callable', *optional*, defaults to '"gelu"'):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string '"gelu"', '"relu"', '"silu"' and '"gelu_new"' are supported
        hidden_dropout_prob ('float', *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob ('float', *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings ('int', *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. 
            Typically set this to something large just in case ( e.g., 512 or 1024 or 2048 ).
        type_vocab_size ('int', *optional*, defaults to 2):
            The vocabulary size of the 'token_type_ids' passed when calling ['QILBertModel']
        initializer_range('float', *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps('float', *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type('str', *optional*, defaults to '"absolute"'):
            Type of the position embedding. Choose one of '"absolute"', '"relative_key"', '"relative_key_query"'.
            For positional embeddings use '"absolute"'. For more information on '"relative_key"',
            please refer to [Self-Attention with Relative Position Representations ( Shaw et al. )]
            (https://arxiv.org/abs/1803.02155).
            For more information on '"relative_key_query"', please refer to *Method 4* in 
            [Improve Transformer Models with Better Relative Position Embeddings ( Huang et al. )]
            (https://arxiv.org/abs/2009.13658).
        quant_mode ('bool', *optional*, defaults to 'False'):
            Whether to quantize the model or not.
    """

    model_type = "QILBert"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        weight_bit = 8,
        act_bit = 8,
        bias_bit = 32,
        quant_mode=False,
        group_wise = False,
        num_groups = 12,
        embedding_bit = 8,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.quant_mode = quant_mode
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.bias_bit = bias_bit
        self.num_labels = 2
        self.group_wise = group_wise
        self.num_groups = num_groups 
        self.embedding_bit = embedding_bit
class IBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1:"sequence"}),
                ("attention_mask", {0:"batch", 1:"sequence"}),
            ]
        )



