""" PyTorch QIL bert model."""
import math
from torch import nn
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from quant_scaling_modules import QuantEmbedding, QILLinear, QILQuantAct, DuQGeLU
import torch
from transformers import AutoTokenizer
from configuration_qilbert import QILBertConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
import time
import matplotlib.pyplot as plt
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.utils import add_start_docstrings
from transformers import AutoModelForSequenceClassification
import numpy as np
import random
torch.set_printoptions(sci_mode = False)
def set_seed(random_seed = 34):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class QILBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.embedding_bit = config.embedding_bit
        self.embedding_act_bit = 32
        self.act_bit = config.act_bit
        self.ln_input_bit = 22
        self.ln_output_bit = 32
        self.debug_mode = False
        self.double_sided = config.double_sided
        self.word_embeddings = QuantEmbedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id,
                weight_bit=self.embedding_bit,
                quant_mode=self.quant_mode,
                double_sided = self.double_sided
        )
        self.token_type_embeddings = QuantEmbedding(
                config.type_vocab_size, config.hidden_size, weight_bit = self.embedding_bit, 
                quant_mode = self.quant_mode,
                double_sided = self.double_sided,
        )

        # position_dis ( 1, len position emb ) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = QuantEmbedding(
                config.max_position_embeddings,
                config.hidden_size,
                padding_idx = self.padding_idx,
                weight_bit = self.embedding_bit,
                quant_mode = self.quant_mode,
                double_sided = self.double_sided,
        )
        
        # Quantization addition btw embeddings
        self.embeddings_act1 = QILQuantAct(self.embedding_act_bit, quant_mode = False)
        self.embeddings_act2 = QILQuantAct(self.embedding_act_bit, quant_mode = False)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load 
        self.LayerNorm = nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
        )

        self.output_activation = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, double_sided = self.double_sided)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self, input_ids = None, token_type_ids = None, position_ids = None, inputs_embeds = None, 
            past_key_values_length = 0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input tokedn ids, Any padded tokens remain padded
                position_ids = create_position_ids_from_input_ids(
                        input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embed(inputs_embeds)
        
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1] # hidden_dim은 뺌 (batch_size, seq_length)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device = self.position_ids.device)
        
        if inputs_embeds is None:
            inputs_embeddings, inputs_embeddings_scaling_factor, inputs_embeddings_offset = \
                    self.word_embeddings(input_ids)
        else:
            inputs_embeddings_scaling_factor = None
        token_type_embeddings, token_type_embeddings_scaling_factor, token_type_embeddings_offset = \
                self.token_type_embeddings(token_type_ids)

        embeddings, embeddings_scaling_factor, embeddings_offset = self.embeddings_act1(
                inputs_embeddings,
                inputs_embeddings_scaling_factor,
                inputs_embeddings_offset,
                identity = token_type_embeddings,
                identity_scaling_factor = token_type_embeddings_scaling_factor,
                identity_offset = token_type_embeddings_offset
        )
        if self.position_embedding_type == "absolute":
            position_embeddings, position_embeddings_scaling_factor, position_embeddings_offset = \
                    self.position_embeddings(position_ids)
            embeddings, embeddings_scaling_factor, embeddings_offset = self.embeddings_act1(
                    embeddings,
                    identity = position_embeddings
            )
        if self.debug_mode:
            print("input")
            print(inputs_embeddings)
            input()
            print("tokentype")
            print(token_type_embeddings)
            input()
            print("position")
            print(position_embeddings)
            input()
            print("embeddings")
            print(embeddings)
            input()
        embeddings = self.LayerNorm(embeddings)
        if self.debug_mode:
            print("after layer norm")
            print(embeddings)
            input()
        embeddings = self.dropout(embeddings)
        embeddings, embeddings_scaling_factor, embeddings_offset = self.output_activation(embeddings)
        if self.debug_mode:
            print("after quant")
            print(embeddings)
            input()
        return embeddings, embeddings_scaling_factor, embeddings_offset

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        we are provided embeddings directly, We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor

        Returns : torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
                self.padding_idx + 1, sequence_length + self.padding_idx +1, dtype = torch.long, 
                device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length = 0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1.
    Padding Symbols are ignored. This is modified from fairseq's *utils.make_positions*

    Args:
    input_ids ('torch.LongTensor'):
        Indices of input sequence tokens in the vocabulary.

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int() # padding idx = 0, otherwise 1
    # 각각의 batch에 대해 mask 위치를 찾아냄
    # cumsum을 통해서 position idx를 알아내고 mask를 곱해서 padding을 날림
    incremental_indicies = (torch.cumsum(mask, dim =1).type_as(mask) + past_key_values_length) * mask
    # 0으로 만든 idx에 padding_idx를 더해서 padding으로 만듦 
    return incremental_indicies.long() + padding_idx


class QILBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                    f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                    f"heads ({num_attention_heads})"
            )
        self.quant_mode = config.quant_mode
        self.weight_bit = config.weight_bit
        self.bias_bit = config.bias_bit
        self.act_bit = config.act_bit

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.double_sided = config.double_sided
        # Q, K, V Linear layers --> channel wise quantization도 고려해볼 만한듯
        self.query = QILLinear(
                config.hidden_size,
                self.all_head_size,
                bias = True,
                weight_bit = self.weight_bit,
                bias_bit = self.bias_bit,
                quant_mode = self.quant_mode,
                per_group = config.group_wise,
                num_groups = config.num_groups,
                double_sided = self.double_sided,
                )
        self.key = QILLinear(
                config.hidden_size,
                self.all_head_size,
                bias = True,
                weight_bit = self.weight_bit,
                bias_bit = self.bias_bit,
                quant_mode = self.quant_mode,
                per_group = config.group_wise,
                num_groups = config.num_groups,
                double_sided = self.double_sided,
                )
        self.value = QILLinear(
                config.hidden_size,
                self.all_head_size,
                bias = True,
                weight_bit = self.weight_bit,
                bias_bit = self.bias_bit,
                quant_mode = self.quant_mode,
                per_group = config.group_wise,
                num_groups = config.num_groups,
                double_sided = self.double_sided,
                )
        
        # Requantization ( 32bit -> 8bit ) for Q, K, V activations
        self.query_activations = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, per_group = config.group_wise, num_groups = config.num_groups, double_sided = self.double_sided)
        self.key_activations = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, per_group = config.group_wise, num_groups = config.num_groups, double_sided = self.double_sided)
        self.value_activations = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, per_group = config.group_wise, num_groups = config.num_groups, double_sided = self.double_sided)
        self.output_activations = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, double_sided = self.double_sided)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.debug_mode = False
        assert (
                self.position_embedding_type == "absolute"
        ), "QIL-BERT only supports 'absolute' for position_embedding_type"

        self.softmax = nn.Softmax(dim = -1)
        #self.after_softmax = QILQuantAct(self.act_bit, quant_mode = self.quant_mode)
    def transpose_for_scores(self, x):
        # ( bs, seq, 768 ) -> ( bs, seq, 12, 64 )
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # ( bs, seq, 12, 64 ) -> ( bs, 12, seq, 64 )
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            hidden_states_scaling_factor,
            hidden_states_offset,
            attention_mask = None,
            head_mask = None,
            output_attentions = False,
    ):
        # Projection
        mixed_query_layer,mixed_query_layer_scaling_factor, mixed_query_layer_offset = \
                self.query(hidden_states, hidden_states_scaling_factor, hidden_states_offset)
        mixed_key_layer, mixed_key_layer_scaling_factor, mixed_key_layer_offset = \
                self.key(hidden_states, hidden_states_scaling_factor, hidden_states_offset)
        mixed_value_layer, mixed_value_layer_scaling_factor, mixed_value_layer_offset = \
                self.value(hidden_states, hidden_states_scaling_factor, hidden_states_offset)
        
        if self.debug_mode:
            print("after query layer")
            print(mixed_query_layer)
            input()
            print("after key layer")
            print(mixed_key_layer)
            input()
            print("after value layer")
            print(mixed_value_layer)
            input()
        # Requantization
        query_layer, query_layer_scaling_factor, query_layer_offset = \
                self.query_activations(mixed_query_layer, mixed_query_layer_scaling_factor, mixed_query_layer_offset)
        key_layer, key_layer_scaling_factor, key_layer_offset = \
                self.key_activations(mixed_key_layer, mixed_key_layer_scaling_factor, mixed_key_layer_offset)
        value_layer, value_layer_scaling_factor, value_layer_offset = \
                self.value_activations(mixed_value_layer, mixed_value_layer_scaling_factor, mixed_value_layer_offset)
        if self.debug_mode:
            '''
            print("after query layer quant")
            print(query_layer)
            #plt.subplot(3, 1, 1)
            #plt.imshow(query_layer[0,:18,:64].clone().cpu().detach().numpy())
            #plt.boxplot(query_layer[0,:18,:64].t().clone().cpu().detach().numpy())
            #plt.figure(figsize = (10,8))
            #plt.subplot(3, 1, 1)
            plt.boxplot(query_layer[0, :18, :].t().clone().cpu().detach().numpy())
            plt.subplot(3, 1, 2)
            plt.boxplot(query_layer.reshape(-1).clone().cpu().detach().numpy())
            plt.subplot(3, 1, 3)
            plt.boxplot([query_layer[0, :18, (i*64):(i*64)+64].reshape(-1).cpu().detach().numpy() for i in range(12)])
            plt.savefig('./act_box/query' +  str(time.time()) + '.png')
            plt.clf()
            input()
            print("after key layer quant")
            print(key_layer)
            #plt.figure(figsize = (10,8))
            #plt.subplot(3, 1, 1)
            #plt.imshow(key_layer[0,:18,:64].clone().cpu().detach().numpy())
            #plt.boxplot(key_layer[0,:18,:64].t().clone().cpu().detach().numpy())
            #plt.boxplot(key_layer[0, :18, :].t().clone().cpu().detach().numpy())
            #plt.subplot(3, 1, 2)
            #plt.boxplot(key_layer.reshape(-1).clone().cpu().detach().numpy())
            #plt.subplot(3, 1, 3)
            #plt.boxplot([key_layer[0, :18, (i*64):(i*64)+64].reshape(-1).cpu().detach().numpy() for i in range(12)])
            #plt.savefig('./act_box/key' +  str(time.time()) + '.png')
            #plt.clf()
            input()
            print("after value layer quant")
            print(value_layer)
            input()
            '''
            plt_list_query = [query_layer[0, :18, (i*64):((i+1)*64)].reshape(-1).detach().cpu().numpy() for i in range(12)]
            plt_list_key = [key_layer[0, :18, (i*64):((i+1)*64)].reshape(-1).detach().cpu().numpy() for i in range(12)]
            plt_list_value = [value_layer[0, :18, (i*64):((i+1)*64)].reshape(-1).detach().cpu().numpy() for i in range(12)]
            plt.figure(figsize = (12,10))
            for i in range(12):
                plt.subplot(3, 4, i+1)
                data_list = [plt_list_query[i], plt_list_key[i], plt_list_value[i]]
                boxes = plt.boxplot(data_list, showmeans = True)
            
                plt.xticks([1, 2, 3], ["Q", "K", "V"])
                for idx, data in enumerate(data_list):
                    plt.text(idx + 1 + 0.3, data.max(), str(data.max())[:5], fontdict={'size':8})
                    plt.text(idx + 1 + 0.3, data.min(), str(data.min())[:5], fontdict={'size':8})
                    plt.text(idx + 1 + 0.3, data.mean(), str(data.mean())[:5], fontdict = {'size':8})
                '''
                for idx, data in enumerate(data_list):
                    plt.scatter(idx+1, data.mean())
                '''
                plt.title('head' + str(i))
            plt.savefig('./layer_box/layer' + str(time.time()) + '.png')
            plt.clf()
        # Transpose
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Take the dot product btw "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.debug_mode:
            print("attention_score")
            print(attention_scores)
            input()
        scale = math.sqrt(self.attention_head_size)
        attention_scores = attention_scores / scale
        if attention_mask is not None:
            # Apply the attention mask is ( precomputed for all layers in QILBertModel forward() function )
            attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        if self.debug_mode:
            '''
            print("attention_probs")
            print(attention_probs)
            #plt.subplot(3, 1, 3)
            #plt.hist(torch.argmax(attention_probs[0,0,:18,:], dim = 1).clone().detach().cpu().numpy(), bins = range(0, 18, 1), align = "left")
            #plt.imshow(attention_probs[0,0,:20, :20].clone().detach().cpu().numpy())
            plt.show()
            plt.clf()
            input()
            '''

        #attention_probs = self.after_softmax(attention_probs)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        # (bs, head_num, seq_len, hidden ) -> (bs, seq_len, head_num, hidden) -->
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (bs, seq_len, all_head_size --> 768 )
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.debug_mode:
            print("context_layer")
            print(context_layer)
            input()
        # requantization : 32-bit -> 8-bit
        context_layer, context_layer_scaling_factor, context_layer_offset = self.output_activations(context_layer)
        if self.debug_mode:
            print("context_layer_after_quant")
            print(context_layer)
            input()

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer, )
        output_scaling_factor = (
                (context_layer_scaling_factor, )
        )
        output_offset = (
                (context_layer_offset, )
        )
        return outputs, output_scaling_factor, output_offset

class QILBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = config.act_bit
        self.weight_bit = config.weight_bit
        self.bias_bit = config.bias_bit
        self.ln_input_bit = 22
        self.ln_output_bit = 32
        self.double_sided = config.double_sided
        self.dense = QILLinear(
            config.hidden_size,
            config.hidden_size,
            bias = True,
            weight_bit = self.weight_bit,
            bias_bit = self.bias_bit,
            quant_mode = self.quant_mode,
            #per_channel = True,
            double_sided = self.double_sided
        )
        self.ln_input_act = QILQuantAct(self.ln_input_bit, quant_mode = False)
        self.LayerNorm = nn.LayerNorm(
                config.hidden_size,
                eps = config.layer_norm_eps
        )
        self.output_activation = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, double_sided = self.double_sided)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.debug_mode = False
    def forward(self, hidden_states, hidden_states_scaling_factor, hidden_states_offset, input_tensor,
            input_tensor_scaling_factor, input_tensor_offset
            ):
        hidden_states,hidden_states_scaling_factor, hidden_states_offset = \
                self.dense(hidden_states, hidden_states_scaling_factor, hidden_states_offset)
        if self.debug_mode:
            print("self output")
            print(hidden_states)
            input()
        hidden_states  = self.dropout(hidden_states)
        hidden_states, hidden_states_scaling_factor, hidden_states_offset = \
                self.ln_input_act(hidden_states, identity = input_tensor)
        hidden_states = self.LayerNorm(hidden_states)
        if self.debug_mode:
            print("after layer norm")
            print(hidden_states)
            input()
        hidden_states, hidden_states_scaling_factor, hidden_states_offset = self.output_activation( hidden_states )
        if self.debug_mode:
            print("after layer norm quant")
            print(hidden_states)
            input()
        return hidden_states, hidden_states_scaling_factor, hidden_states_offset

class QILBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.self = QILBertSelfAttention(config)
        self.output = QILBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
                heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim = 1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        hidden_states_offset,
        attention_mask = None,
        head_mask = None,
        output_attention = False,
    ):
        self_outputs, self_outputs_scaling_factor, self_outputs_offset = self.self(
                hidden_states,
                hidden_states_scaling_factor,
                hidden_states_offset,
                attention_mask,
                head_mask,
                output_attention,
        )
        attention_output, attention_output_scaling_factor, attention_output_offset = self.output(
                self_outputs[0], self_outputs_scaling_factor[0], self_outputs_offset[0], 
                hidden_states, hidden_states_scaling_factor, hidden_states_offset,
        )
        outputs = (attention_output,) + self_outputs[1:] # add attentions if we output them
        outputs_scaling_factor = (attention_output_scaling_factor,) + self_outputs_scaling_factor[1:]
        outputs_offset = (attention_output_offset,) + self_outputs_offset[1:]
        return outputs, outputs_scaling_factor, outputs_offset

class QILBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = config.act_bit
        self.weight_bit = config.weight_bit
        self.bias_bit = config.bias_bit
        self.double_sided = config.double_sided
        self.dense = QILLinear(
                config.hidden_size,
                config.intermediate_size,
                bias = True,
                weight_bit = self.weight_bit,
                bias_bit = self.bias_bit,
                quant_mode = self.quant_mode,
                #per_channel = True,
                double_sided = self.double_sided,
        )
        self.intermediate_act_fn = nn.GELU()
        self.before_activation_quant = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, double_sided = self.double_sided)
        self.output_activation = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, double_sided = self.double_sided)
        self.debug_mode = False

    def forward(self, hidden_states, hidden_states_scaling_factor, hidden_states_offset):
        hidden_states,hidden_states_scaling_factor, hidden_states_offset = \
                self.dense(hidden_states, hidden_states_scaling_factor, hidden_states_offset)
        if self.debug_mode:
            print("intermediate dense")
            print(hidden_states)
            input()
        '''
        hidden_states, hidden_states_scaling_factor, hidden_states_offset = self.before_activation_quant(hidden_states)
        if self.debug_mode:
            print("quant before gelu")
            print(hidden_states)
            input()
        '''
        hidden_states = self.intermediate_act_fn(hidden_states)
        if self.debug_mode:
            print("gelu")
            print(hidden_states)
            input()

        # Requantization : 32bit -> 8bit
        hidden_states, hidden_states_scaling_factor, hidden_states_offset = self.output_activation(hidden_states, 
                hidden_states_scaling_factor, hidden_states_offset)

        return hidden_states, hidden_states_scaling_factor, hidden_states_offset

class QILBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = config.act_bit
        self.weight_bit = config.weight_bit
        self.bias_bit = config.bias_bit
        self.ln_input_bit = 22
        self.ln_output_bit = 32
        self.double_sided = config.double_sided
        self.dense = QILLinear(
                config.intermediate_size,
                config.hidden_size,
                bias = True,
                weight_bit = self.weight_bit,
                bias_bit = self.bias_bit,
                quant_mode = self.quant_mode,
                double_sided = self.double_sided,
                #per_channel = True,
        )
        self.ln_input_act = QILQuantAct(self.ln_input_bit, quant_mode = False)
        self.LayerNorm = nn.LayerNorm(
                config.hidden_size,
                eps = config.layer_norm_eps,
                )
        self.output_activation = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, double_sided = self.double_sided)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.debug_mode = False

    def forward(self, hidden_states, hidden_states_scaling_factor, hidden_states_offset,
            input_tensor, input_tensor_scaling_factor, input_tensor_offset):
        hidden_states,hidden_states_scaling_factor, hidden_states_offset = \
                self.dense(hidden_states,hidden_states_scaling_factor, hidden_states_offset)
        if self.debug_mode:
            print("output dense")
            print(hidden_states)
            input()
        hidden_states = self.dropout(hidden_states)
        hidden_states, hidden_states_scaling_factor, hidden_states_offset = \
                self.ln_input_act(hidden_states, hidden_states_scaling_factor, hidden_states_offset,
                        identity = input_tensor, identity_scaling_factor = input_tensor_scaling_factor, 
                        identity_offset = input_tensor_offset)
        
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, hidden_states_scaling_factor, hidden_states_offset = \
                self.output_activation(hidden_states, hidden_states_scaling_factor)
        if self.debug_mode:
            print("output act")
            print(hidden_states)
            input()
        return hidden_states, hidden_states_scaling_factor, hidden_states_offset

class QILBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.act_bit = config.act_bit
        
        self.seq_len_dim = 1
        self.attention = QILBertAttention(config)
        self.intermediate = QILBertIntermediate(config)
        self.output = QILBertOutput(config)
        self.double_sided = config.double_sided
        self.pre_intermediate_act = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, double_sided = self.double_sided)
        self.pre_output_act = QILQuantAct(self.act_bit, quant_mode = self.quant_mode, double_sided = self.double_sided)

    def forward(
            self,
            hidden_states,
            hidden_states_scaling_factor,
            hidden_states_offset,
            attention_mask = None,
            head_mask = None,
            output_attentions = False,
    ):
        self_attention_outputs, self_attention_outputs_scaling_factor, self_attention_outputs_offset = \
        self.attention(
                hidden_states,
                hidden_states_scaling_factor,
                hidden_states_offset,
                attention_mask,
                head_mask,
                output_attention = output_attentions
        )
        attention_output = self_attention_outputs[0]
        attention_output_scaling_factor = self_attention_outputs_scaling_factor[0]
        attention_output_offset = self_attention_outputs_offset[0]

        outputs = self_attention_outputs[1:] # add self attentions if we output attention weights

        layer_output, layer_output_scaling_factor, layer_output_offset = self.feed_forward_chunk(
                attention_output, attention_output_scaling_factor, attention_output_offset
                )
        outputs = (layer_output, ) + outputs
        return outputs
    
    def feed_forward_chunk(self, attention_output, attention_output_scaling_factor, attention_output_offset):
        attention_output,attention_output_scaling_factor, attention_output_offset = \
                self.pre_intermediate_act(attention_output, attention_output_scaling_factor, attention_output_offset)
        intermediate_output, intermediate_output_scaling_factor, intermediate_output_offset = \
                self.intermediate(attention_output, attention_output_scaling_factor, attention_output_offset)
        intermediate_output, intermediate_output_scaling_factor, intermediate_output_offset =\
                self.pre_output_act(intermediate_output, intermediate_output_scaling_factor, intermediate_output_offset)
        layer_output, layer_output_scaling_factor, layer_output_offset = \
            self.output(intermediate_output, intermediate_output_scaling_factor, intermediate_output_offset,
                    attention_output, attention_output_scaling_factor, attention_output_offset)
        return layer_output, layer_output_scaling_factor, layer_output_offset

class QILBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quant_mode = config.quant_mode
        self.layer = nn.ModuleList([QILBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            hidden_states_scaling_factor,
            hidden_states_offset,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None # 'config.add_cross_attention' is not supported
        next_decoder_cache = None  # 'config.use_cache' is not supported
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                    hidden_states,
                    hidden_states_scaling_factor,
                    hidden_states_offset,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_decoder_cache,
                        all_hidden_states,
                        all_self_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
        return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state = hidden_states,
                past_key_values = next_decoder_cache,
                hidden_states = all_hidden_states,
                attentions = all_self_attentions,
                cross_attentions = all_cross_attentions,
        )
# Pooler에는 Quantization 적용하지 않음
class QILBertPooler(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.quant_mode = config.quant_mode
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class QILBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and 
    loading pretrained models
    """
    config_class = QILBertConfig
    base_model_prefix = "qilbert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (QILLinear, nn.Linear)):
            module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (QuantEmbedding, nn.Embedding)):
            module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def resize_token_embeddings(self, new_num_tokens = None):
        raise NotImplementedError("'resize_token_embeddings' is not supported for QIL-BERT.")

@add_start_docstrings(
        "The bare QILBert Model transformer outputting raw hidden-states without any specific head on top.",
)

class QILBertModel(QILBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a 
    layer of cross-attention is added between the self-attention layers, following the architecture
    described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer = True):
        super().__init__(config)
        self.config = config
        self.quant_mode = config.quant_mode

        self.embeddings = QILBertEmbeddings(config)
        self.encoder = QILBertEncoder(config)

        self.pooler = QILBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing

        self.post_init()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ 
        Prunes heads of the model. heads_to_prune : dict of { layer_num : list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask : Optional[torch.FloatTensor] = None,
            token_type_ids : Optional[torch.LongTensor] = None,
            position_ids : Optional[torch.LongTensor] = None,
            head_mask : Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions : Optional[bool] = None,
            output_hidden_states : Optional[bool] = None,
            return_dict : Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, Tuple[torch.FloatTensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not NOne:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device = device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device = device)

        # we can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, embedding_output_scaling_factor, embedding_output_offset = self.embeddings(
                input_ids = input_ids,
                position_ids = position_ids,
                token_type_ids = token_type_ids,
                inputs_embeds = inputs_embeds,
        )
        encoder_outputs = self.encoder(
                embedding_output,
                embedding_output_scaling_factor,
                embedding_output_offset,
                attention_mask = extended_attention_mask,
                head_mask = head_mask,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state = sequence_output,
                pooler_output = pooled_output,
                past_key_values = encoder_outputs.past_key_values,
                hidden_states = encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
        )

class QILBertClassificationHead(nn.Module):
    """ Head for sentence-level classification tasks."""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
                                
    def forward(self, features, **kwargs):
        hidden_states = features[:, 0, :] # tkae <CLS> token ( equiv. to <s> )
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class QILBertForSequenceClassification(QILBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.qilbert = QILBertModel(config, add_pooling_layer = False)
        self.classifier = QILBertClassificationHead(config)

        self.post_init()

    def forward(
            self,
            input_ids : Optional[torch.LongTensor] = None,
            attention_mask : Optional[torch.FloatTensor] = None,
            token_type_ids : Optional[torch.LongTensor] = None,
            position_ids : Optional[torch.LongTensor] = None,
            head_mask : Optional[torch.FloatTensor] = None,
            inputs_embeds : Optional[torch.FloatTensor] = None,
            labels : Optional[torch.LongTensor] = None,
            output_attentions : Optional[bool] = None,
            output_hidden_states : Optional[bool] = None,
            return_dict : Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels ('torch.LongTensor' of shape '(batch_size,)', *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in '[0, ...,
            config.num_labels - 1]'. If 'config.num_labels == 1' a regression loss is computed ( MSE ), 
            If config.num_labels > 1' a classification loss is computed ( Cross Entropy loss ).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.qilbert(
                input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                head_mask = head_mask,
                inputs_embeds = inputs_embeds,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        print(logits)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
                loss = loss,
                logits = logits,
                hidden_states = outputs.hidden_states,
                attentions = outputs.attentions,
        )

if __name__ == "__main__":
    set_seed()
    config = QILBertConfig()
    config.update({'quant_mode' : True})
    embedding = QILBertEmbeddings(config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    encoded = tokenizer("hello, my name is Lee Sang Ho", return_tensors = "pt")
    embedded, embedded_scaling_factor, embedded_offset = embedding(encoded['input_ids'])
    print(embedded, embedded_scaling_factor, embedded_offset)
    self_attention = QILBertForSequenceClassification(config)
    print(self_attention(encoded['input_ids']))
    #quant = embedding(encoded['input_ids'])
    
    #Encoder = QILBertEncoder(config)
    #Inter = QILBertIntermediate(config)
    #classification = QILBertForSequenceClassification.from_pretrained('./QILBert_pretrained')
    #classification.eval()
    #roberta_pretrained = AutoModelForSequenceClassification.from_pretrained('roberta-base')
    #roberta_pretrained.eval()
    '''
    state_dict = roberta_pretrained.state_dict()
    new_state_dict = {}
    for (key, value) in state_dict.items():
        if "roberta" in key:
            new_key = key.replace('roberta', 'qilbert')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    torch.save(new_state_dict, './QILBert_pretrained/pytorch_model.bin')
    classification.load_state_dict(new_state_dict, strict = True)
    classification.save_pretrained('./QILBert_pretrained')
    '''
    
