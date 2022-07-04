import decimal

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
#from ...utils import logging

#logger = logging.get_logger(__name__)

class round_ste(Function):
    """
    STE for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class floor_ste(Function):
    """
    STE for torch.floor()
    """
    
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class QuantEmbedding(nn.Module):
    """
    Quantized version of 'torch.nn.Embedding'. Adds quantization-specific arguments on top of 'torch.nn.Embedding'.

    Args :
        weight_bit ('int', *optional*, defaults to '8'):
            Bitwidth for the quantized weight.
        momentum ('float', *optional*, defautls to '0.95'):
            Momentum for updating the activation quantization range ?
        quant_mode ('bool', *optional*, defaults to 'False'):
            Wheather or not the layer is quantized
    """
    def __init__(
            self, 
            num_embeddings,
            embedding_dim,
            padding_idx = None,
            max_norm = None,
            norm_type = 2.0,
            scale_grad_by_freq = False,
            sparse = False,
            _weight = None,
            weight_bit = 8,
            momentum = 0.95,
            quant_mode = False,
    ):
        super().__init__()
        self.num_ = num_embeddings
        self.dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.momentum = momentum
        if _weight is None:
            self.weight = nn.Parameter(torch.randn([num_embeddings, embedding_dim]))
        else:
            self.weight = _weight

        self.quant_mode = quant_mode
        #self.weight_function = QILQuantFunction.apply 
        self.weight_discretizer = QILWeightDiscretizer.apply

        self.bitW = weight_bit
        self.c_W = nn.Parameter(torch.ones(2))
        self.d_W = nn.Parameter(torch.ones(2))
        self.gamma = nn.Parameter(torch.ones(1))
        self.register_buffer("weight_quantize", torch.zeros_like(self.weight))
        self.register_buffer("alpha_W", torch.zeros(1))
        self.register_buffer("beta_W", torch.zeros(1))
        self.register_buffer("weight_scaling_factor", torch.zeros(1))
        self.register_buffer("weight_offset", torch.zeros(1))
        self.epsilon = 1e-12
        self.weight_module_init = True
        self.act_init_mode = False
        if self.quant_mode == True:
            print('QIL Embedding Quantization Initialization with %d-bit.' %self.bitW)
        else:
            print('fine-tuning with full-precision model')

    def forward(self, x, positions = None, incremental_state = None):
        if not self.quant_mode :
            return nn.functional.embedding(
                    x,
                    self.weight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                ), None, None
        if self.d_W > self.c_W:
            self.d_W.data = self.c_W.data

        assert self.c_W >= 0, "center parameter not be zero"
        #transform
        data_type = torch.FloatTensor if x.device == torch.device("cpu")  else torch.cuda.FloatTensor
        self.alpha_W = 0.5 / (self.d_W )
        self.beta_W = -0.5 * self.c_W / (self.d_W ) + 0.5
        
        self.weight_quantize = (self.alpha_W * torch.abs(self.weight) + self.beta_W)
        self.weight_quantize = torch.clamp(self.weight_quantize, min = self.epsilon, max = 1.)
        #self.weight_quantize = torch.pow(self.weight_quantize, self.gamma) * torch.sign(self.weight)
        self.weight_quantize = torch.where(self.weight_quantize > self.epsilon, self.weight_quantize ** self.gamma, self.weight_quantize * 0) 
        self.weight_quantize = self.weight_discretizer(self.weight_quantize, self.bitW) * torch.sign(self.weight)
        self.weight_offset = self.c_W - self.d_W
        self.weight_scaling_factor = 2 * self.d_W
        embed_quantize = nn.functional.embedding(
                input = x,
                weight = self.weight_quantize,
                padding_idx = self.padding_idx,
                max_norm = self.max_norm,
                norm_type =self.norm_type,
                scale_grad_by_freq = self.scale_grad_by_freq,
                sparse = self.sparse,
        )
        return (torch.abs(embed_quantize) * self.weight_scaling_factor + self.weight_offset) *\
                torch.sign(embed_quantize), self.weight_scaling_factor, self.weight_offset
    
    def quantize_reset_parameter(self, args):
        w_abs = self.weight.abs().clone().detach()
        if args.weight_init_mode == "min_max":
            w_max = w_abs.max()
            w_min = w_abs.min()
            w_dx = (w_max + w_min).data / 2
            w_dy = (w_max - w_min).data / 2
        elif args.weight_init_mode == "topk":
            w_abs = w_abs.reshape(-1).sort()[0]
            w_top = w_abs[int(len(w_abs) * 0.7)].data
            w_bottom = w_abs[0].data

            w_dx = (w_top + w_bottom) / 2
            w_dy = (w_top - w_bottom) / 2
        else:
            w_mean = w_abs.mean()
            w_std = w_abs.std()
            w_dx = w_mean.data
            w_dy = w_std.data
        self.c_W.data = torch.Tensor([w_dx]).to(self.weight.device)
        self.d_W.data = torch.Tensor([w_dy]).to(self.weight.device)
    
    def box_plot(self, name):
        w_abs = self.weight.abs().reshape(-1).sort()[0].clone().detach().cpu().numpy()
        plt.boxplot(w_abs)
        plt.hlines(w_abs[int(len(w_abs) * 0.7)], 0.75, 1.25)
        plt.hlines(w_abs[int(len(w_abs) * 0.9)], 0.75, 1.25)
        plt.hlines(w_abs[int(len(w_abs) * 0.95)], 0.75, 1.25)
        plt.title(name)
        plt.savefig('./fig_box/' + name + '.png')
        plt.clf()
    def quantize_reset_parameter_other(self):
        w_abs = self.weight.abs().clone().detach()
        w_mean = w_abs.mean()
        w_std = w_abs.std()

        w_dx = w_mean.data
        w_dy = w_std.data

        self.c_W.data = torch.Tensor([w_dx]).to(self.weight.device)
        self.d_W.data = torch.Tensor([w_dy]).to(self.weight.device)

        '''
        w = self.weight.clone().detach().reshape(-1)
        n = len(w)
        w_abs = w.abs()
        #w = w.sort()[0]
        w_abs_sort = w_abs.sort(descending = True)[0]
        '''
        '''
        index_min = int(percentage * n / 2)
        index_max = n-index_min

        min_value = abs(w[index_min-1].data)
        max_value = abs(w[index_max-1].data)
        
        if min_value >= max_value:
            self.c_W.data = torch.Tensor([min_value / 2]).to(self.weight.device)
        else:
            self.c_W.data = torch.Tensor([max_value / 2]).to(self.weight.device)
        self.d_W.data = self.c_W.data
        '''
        '''
        abs_index = int(n * (percentage))
        self.c_W.data = torch.Tensor([w_abs_sort[abs_index].data / 2]).to(self.weight.device)
        self.d_W.data = self.c_W.data
        '''
    def weight_plot(self, name):
        w = self.weight.clone().cpu().detach().reshape(-1).numpy()

        plt.figure(figsize = (10,8))
        plt.subplot(3, 1, 1)
        plt.hist(w, bins = 10000)
        plt.xlim(-np.max(np.abs(w)), np.max(np.abs(w)))
        plt.axvline(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy()), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy()), color = "red")
        w_quant = (self.alpha_W * torch.abs(self.weight) + self.beta_W)
        w_quant = torch.clamp(w_quant, min = self.epsilon, max = 1.)
        w_quant = torch.where(w_quant > self.epsilon, w_quant ** self.gamma, w_quant * 0 )
        w_quant = self.weight_discretizer(w_quant, self.bitW) * torch.sign(self.weight)
        w_offset = self.c_W - self.d_W
        w_weight_scaling_factor = 2 * self.d_W
        w_quant = (torch.abs(w_quant) * w_weight_scaling_factor + w_offset) * torch.sign(w_quant)
        w_quant = w_quant.clone().cpu().detach().reshape(-1).numpy()
        w_pru = torch.where(self.weight.abs() < self.c_W - self.d_W, 1, 0).reshape(-1)
        w_cli = torch.where(self.weight.abs() > self.c_W + self.d_W, 1, 0).reshape(-1)
        font = {
                'fontsize' : 8,
        }
        plt.title("name : " + name + "\npruning : " + str(w_pru.sum().data / len(w)) + "\nclipping : " + str(w_cli.sum().data / len(w)), fontdict = font)
        plt.subplot(3, 1, 2)
        plt.hist(w_quant, bins = 256)
        plt.xlim(-np.max(np.abs(w)), np.max(np.abs(w)))
        plt.axvline(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy()), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy()), color = "red")
        
        plt.subplot(3, 1, 3)
        plt.xlim(-np.max(np.abs(w)), np.max(np.abs(w)))
        plt.hist(w, bins =10000)
        plt.hist(w_quant, bins = 256)
        plt.axvline(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy()), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy()), color = "red")
        
        print(name, len(w), w_pru.sum().data / len(w), w_cli.sum().data / len(w))
        #plt.savefig('./fig/' + name + '.png')
        #plt.clf()

    def print_percentage(self, name):
        w = self.weight.clone().cpu().detach().reshape(-1).numpy()
        w_quant = (self.alpha_W * torch.abs(self.weight) + self.beta_W)
        w_quant = torch.clamp(w_quant, min = self.epsilon, max = 1.)
        w_quant = torch.where(w_quant > self.epsilon, w_quant ** self.gamma, w_quant * 0 )
        w_quant = self.weight_discretizer(w_quant, self.bitW) * torch.sign(self.weight)
        w_offset = self.c_W - self.d_W
        w_weight_scaling_factor = 2 * self.d_W
        w_quant = (torch.abs(w_quant) * w_weight_scaling_factor + w_offset) * torch.sign(w_quant)
        w_quant = w_quant.clone().cpu().detach().reshape(-1).numpy()
        w_pru = torch.where(self.weight.abs() < self.c_W - self.d_W, 1, 0).reshape(-1)
        w_cli = torch.where(self.weight.abs() > self.c_W + self.d_W, 1, 0).reshape(-1)
        return len(w), w_pru, w_cli

class QILActDiscretizer(Function):
    @staticmethod
    def forward(ctx, x, bit):
        n = float(2 ** (bit-1) -1)
        scaling_factor = 1 / n
        return torch.round(x / scaling_factor) * scaling_factor

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone(), None

class QILWeightDiscretizer(Function):
    @staticmethod
    def forward(ctx, x, bit):
        n = float(2 ** (bit-1) -1)
        scaling_factor = 1 / n
        return torch.round(x / scaling_factor) * scaling_factor

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone(), None

class DuQActDiscretizer(Function):
    @staticmethod
    def forward(ctx, x, bit):
        n = float(2**bit - 1)
        scaling_factor = 1 / n
        return torch.round(x / scaling_factor) * scaling_factor
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone(), None

class DuQGeLU(nn.Module):
    def __init__(self, bitA = 8, quant_mode = False):
        super().__init__()

        self.quant_mode = quant_mode
        self.bitA = bitA
        self.act_discritizer = DuQActDiscretizer.apply

        self.c_A = nn.Parameter(torch.Tensor([1.]))
        self.d_A = nn.Parameter(torch.Tensor([1.]))
        self.gamma  = nn.Parameter(torch.Tensor([2.]))

        self.epsilon = 1e-12
        self.offset = 0.17
        self.register_buffer('alpha_A', torch.zeros(1))
        self.register_buffer('beta_A', torch.zeros(1))
        
        self.act_module_init = True
        self.act_init_mode = False

        if self.quant_mode == True:
            print('DuQ GeLU Initialization')
        else:
            print('GeLU Init')
    def forward(self, x):
        if not self.quant_mode:
            return nn.functional.gelu(x)
        self.alpha_A = 0.5 / self.d_A
        self.beta_A = -0.5 * self.c_A / self.d_A + 0.5
        quantized_act = nn.functional.gelu(x * self.gamma)
        quantized_act = quantized_act + self.offset
        quantized_act = (self.alpha_A * quantized_act + self.beta_A)
        quantized_act = torch.clamp(quantized_act, min = 0., max = 1.)
        
        quantized_act = self.act_discritizer(quantized_act, self.bitA)
        quantized_act = quantized_act - self.offset/self.gamma
        
        return quantized_act

class QILQuantAct(nn.Module):
    def __init__(self, bitA = 8, quant_mode = False, act_range_momentum = 0.95, per_group = False, num_groups = 12,channel_len = None):
        super().__init__()

        self.quant_mode = quant_mode
        self.act_range_momentum = act_range_momentum
        self.per_group = per_group
        self.num_groups = num_groups
        self.percentile = False

        self.bitA = bitA
        self.act_discritizer = QILActDiscretizer.apply

        
        self.percentage = 0.01
        self.epsilon = 1e-12
 
        if self.per_group:
            self.c_A = nn.Parameter(torch.zeros([num_groups, 1]))
            self.d_A = nn.Parameter(torch.zeros([num_groups, 1]))

        else:
            self.c_A = nn.Parameter(torch.zeros(1))
            self.d_A = nn.Parameter(torch.zeros(1))
        
        self.register_buffer("act_scaling_factor", torch.zeros_like(self.c_A))
        self.register_buffer("act_offset", torch.zeros_like(self.c_A))
        self.register_buffer("alpha_A", torch.zeros_like(self.c_A))
        self.register_buffer("beta_A", torch.zeros_like(self.c_A))

        if self.quant_mode == True:
            print('QIL Activation Quantization Initialization with %d-bit.' %self.bitA)
        else:
            print('fine-tuning with full-precision module')
        self.act_module_init = True
        self.act_init_mode = False
        self.init_mode = "min_max"

    def forward(
        self,
        x,
        pre_act_scaling_factor = None,
        pre_act_offset = None,
        identity = None,
        identity_scaling_factor = None,
        identity_offset = None,
        specified_center = None,
        specified_distance = None,
        ):

        data_type = torch.FloatTensor if x.device == torch.device("cpu") else torch.cuda.FloatTensor
        x_act = x if identity is None else identity + x
        if self.per_group:
            group_shape = x_act.shape[:-1] + (self.num_groups, x_act.shape[-1] // self.num_groups)
        original_shape = x_act.shape

        if self.act_init_mode:
            x_abs = x_act.clone().detach().abs()
            if self.per_group:
                assert x_act.shape[-1] % self.num_groups == 0
                if len(x_act.shape) == 2:
                    argshape = (0, -1)
                elif len(x_act.shape) == 3:
                    argshape = (0, 1, -1)
                else :
                    raise NotImplementedError("input shape need to be 3-dimension or 2-dimension")
                x_abs = x_abs.reshape(group_shape)
                x_max = x_abs.amax(dim = argshape).reshape(self.num_groups, 1)
                x_min = x_abs.amin(dim = argshape).reshape(self.num_groups, 1)
                x_dx = (x_max + x_min).data / 2
                x_dy = (x_max - x_min).data / 2
            else:
                if self.init_mode == "min_max":
                    x_max = x_abs.max()
                    x_min = x_abs.min()
                    x_dx = (x_max + x_min).data /2 
                    x_dy = (x_max - x_min).data /2
                elif self.init_mode == "topk":
                    x_abs = x_abs.reshape(-1)
                    x_abs = x_abs.sort()[0]
                    x_top = x_abs[int(len(x_abs) * 0.95)].data
                    x_bottom = x_abs[0].data
                    x_dx = (x_top + x_bottom) / 2
                    x_dy = (x_top - x_bottom) / 2
                else:
                    x_mean = x_abs.mean()
                    x_std = x_abs.std()
                    x_dx = x_mean.data
                    x_dy = x_std.data
            self.c_A.data = ( 1 - self.act_range_momentum ) * self.c_A.data + self.act_range_momentum * x_dx.data
            self.d_A.data = ( 1 - self.act_range_momentum ) * self.d_A.data + self.act_range_momentum * x_dy.data
    
        if not self.quant_mode:
            return x_act, None, None
        self.d_A.data[self.d_A > self.c_A] = self.c_A.data[self.d_A > self.c_A]
        assert torch.any(self.c_A >= 0), "center parameter not be zero"

        self.alpha_A = 0.5 / (self.d_A + self.epsilon)
        self.beta_A = -0.5 * self.c_A / (self.d_A +self.epsilon)+ 0.5
        self.act_offset = self.c_A - self.d_A
        self.act_scaling_factor = 2 * self.d_A
        if self.per_group:
            x_act = x_act.reshape(group_shape)
        quantized_act = (self.alpha_A * torch.abs(x_act) + self.beta_A)
        quantized_act = torch.clamp(quantized_act, min = 0., max = 1.)
        quantized_act = self.act_discritizer(quantized_act, self.bitA) * torch.sign(x_act)
        '''
        if pre_act_scaling_factor is None:
            # input quantization
            quantized_act = self.act_discritizer(quantized_act, self.bitA)
        else:
            quantized_act = FixedPointMul.apply(
                    x, 
                    pre_act_scaling_factor,
                    self.bitA,
                    self.act_scaling_factor,
                    identity,
                    identity_scaling_factor,
                    )
        '''
        quantized_act = (torch.abs(quantized_act) * self.act_scaling_factor + self.act_offset) * torch.sign(quantized_act)
        if self.per_group:
            quantized_act = quantized_act.reshape(original_shape)
        return quantized_act, self.act_scaling_factor, self.act_offset

class QILLinear(nn.Module):
    """
    QIL Quantized version of 'torch.nn.Linear'. Adds quantization-specific arguments on top of 'torch.nn.Linear'.

    Args:
        weight_bit ('int', *optional*, defaults to '8'):
            Bitwdith for the quantized weight.
        bias_bit ('int', *optional*, defaults to '32'):
            Bitwidth for the quantized bias.
        quant_mode ('bool', *optional*, defaults to 'False'):
            Whether or not the layer is quantized.
    """
    def __init__(
            self, in_features, out_features, bias = True, weight_bit = 8, bias_bit = 32, quant_mode = False, per_group = False, num_groups = 12
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.per_group = per_group
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.rand([out_features, in_features]))
        self.register_buffer("weight_quantize", torch.zeros_like(self.weight))

        if self.per_group:
            assert out_features % num_groups == 0
            self.c_W = nn.Parameter(torch.ones([num_groups, 1, 1]))
            self.d_W = nn.Parameter(torch.ones([num_groups, 1, 1]))
        else:
            self.c_W = nn.Parameter(torch.ones(1))
            self.d_W = nn.Parameter(torch.ones(1))

        self.register_buffer("beta_W", torch.zeros_like(self.c_W))
        self.register_buffer("alpha_W", torch.zeros_like(self.c_W))
        self.register_buffer("weight_offset", torch.zeros_like(self.c_W))
        self.register_buffer("weight_scaling_factor", torch.zeros_like(self.c_W))
        self.gamma = nn.Parameter(torch.ones_like(self.c_W))

        if bias:
            self.bias = nn.Parameter(torch.zeros([out_features]))
            self.register_buffer("bias_quantize", torch.zeros_like(self.bias))
            self.register_buffer("bias_scaling_factor", torch.zeros(1))
            self.register_buffer("bias_offset", torch.zeros(1))

        self.bitW = weight_bit
        self.quant_mode = quant_mode
        self.bias_bit = bias_bit
        self.weight_discretizer = QILWeightDiscretizer.apply
        self.epsilon = 1e-12
        self.weight_module_init = True
        self.act_init_mode = False

    def forward(self, x, prev_act_scaling_factor = None, prev_act_offset = None):
        if not self.quant_mode:
            return nn.functional.linear(x, weight = self.weight, bias = self.bias), None, None

        data_type = torch.FloatTensor if x.device == torch.device("cpu") else torch.cuda.FloatTensor
        
        self.d_W.data[self.d_W.data > self.c_W.data] = self.c_W.data[self.d_W.data > self.c_W.data]
        self.d_W.data = torch.where(self.d_W.data > self.c_W.data, self.c_W.data, self.d_W.data)

        assert torch.all(self.c_W >= 0), "center parameter not be zero"
        
        # 여기까지는 괜찮을듯 ..?
        self.alpha_W = 0.5 / self.d_W
        self.beta_W = -0.5 * self.c_W / self.d_W  + 0.5
        self.weight_offset = self.c_W - self.d_W 
        self.weight_scaling_factor = 2 * self.d_W
        if self.per_group:
            self.weight_quantize = self.weight.reshape(self.num_groups,self.out_features // self.num_groups, self.in_features)
            self.weight_quantize = (self.alpha_W * torch.abs(self.weight_quantize) + self.beta_W)
        else:
            self.weight_quantize = (self.alpha_W * torch.abs(self.weight) + self.beta_W)
        self.weight_quantize = torch.clamp(self.weight_quantize, min = self.epsilon, max = 1.)
        self.weight_quantize = torch.where(self.weight_quantize > self.epsilon, self.weight_quantize ** self.gamma, \
                self.weight_quantize * 0) 
        self.weight_quantize = self.weight_discretizer(self.weight_quantize, self.bitW) * torch.sign(self.weight).reshape(self.weight_quantize.shape)

        #x_quantize = (( x.abs() - prev_act_offset ) / prev_act_scaling_factor) * torch.sign(x)
        ''' 여기 뜯어 고쳐야함 
        '''
        '''
        if self.bias is not None:
            self.bias_quantize = ( self.bias.abs()- self.bias_offset ) / self.bias_scaling_factor
            self.bias_quantize = self.weight_discretizer(self.bias_quantize, self.bias_bit)
        '''
        self.weight_quantize = (self.weight_quantize.abs() * self.weight_scaling_factor + self.weight_offset) * torch.sign(self.weight_quantize)
        if self.per_group:
            self.weight_quantize = self.weight_quantize.reshape(self.weight.shape)
        return nn.functional.linear(x, weight = self.weight_quantize, bias = self.bias), None, None


    def quantize_reset_parameter(self,args):
        w_abs = self.weight.abs().clone().detach()
        if self.per_group:
            w_abs = w_abs.reshape(self.num_groups, -1)   
            w_max = w_abs.max(dim = -1, keepdim = True)[0]
            w_min = w_abs.min(dim = -1, keepdim = True)[0]
            w_dx = (w_max + w_min).unsqueeze(-1).data / 2
            w_dy = (w_max - w_min).unsqueeze(-1).data / 2
        else:
            if args.weight_init_mode == "min_max":
                w_max = w_abs.max()
                w_min = w_abs.min()
                w_dx = torch.Tensor([(w_max + w_min).data / 2])
                w_dy = torch.Tensor([(w_max - w_min).data / 2])
            elif args.weight_init_mode == "topk":
                w_abs = w_abs.reshape(-1)
                w_abs = w_abs.sort()[0]
                w_top = w_abs[int(len(w_abs) * 0.7)].data
                w_bottom = w_abs[0].data

                w_dx = torch.Tensor([(w_top + w_bottom) / 2])
                w_dy = torch.Tensor([(w_top - w_bottom) / 2])
            else:
                w_mean = w_abs.mean()
                w_std = w_abs.std()
                w_dx = w_mean.data
                w_dy = w_std.data
        self.c_W.data = w_dx.to(self.weight.device).data
        self.d_W.data = w_dy.to(self.weight.device).data
        print(self.c_W, self.d_W)

        '''
        w = self.weight.clone().detach().reshape(-1)
        w_abs = w.abs()
        n = len(w)
        #w = w.sort()[0]
        w_abs_sort = w_abs.sort(descending = True)[0]
        '''
        '''
        index_min = int(percentage * n / 2)
        index_max = n-index_min
        min_value = abs(w[index_min-1].data)
        max_value = abs(w[index_max-1].data)
        if min_value >= max_value:
            self.c_W.data = torch.Tensor([min_value / 2]).to(self.weight.device)
        else:
            self.c_W.data = torch.Tensor([max_value / 2]).to(self.weight.device)
        '''
        '''

        abs_index = int(n * percentage)
        self.c_W.data = torch.Tensor([w_abs_sort[abs_index].data / 2]).to(self.weight.device)
        self.d_W.data = self.c_W.data
        '''
    def weight_plot(self, name):
        w = self.weight.clone().cpu().detach().reshape(-1).numpy()
        plt.figure(figsize = (10,8))
        plt.subplot(3, 1, 1)
        plt.hist(w, bins = 10000)
        plt.xlim(-np.max(np.abs(w)), np.max(np.abs(w)))
        plt.axvline(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy()), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy()), color = "red")
        w_quant = (self.alpha_W * torch.abs(self.weight) + self.beta_W)
        w_quant = torch.clamp(w_quant, min = self.epsilon, max = 1.)
        w_quant = torch.where(w_quant > self.epsilon, w_quant ** self.gamma, w_quant * 0 )
        w_quant = self.weight_discretizer(w_quant, self.bitW) * torch.sign(self.weight)
        w_offset = self.c_W - self.d_W
        w_weight_scaling_factor = 2 * self.d_W
        w_quant = (torch.abs(w_quant) * w_weight_scaling_factor + w_offset) * torch.sign(w_quant)
        w_quant = w_quant.clone().cpu().detach().reshape(-1).numpy()
        w_pru = torch.where(self.weight.abs() < self.c_W - self.d_W, 1, 0).reshape(-1)
        w_cli = torch.where(self.weight.abs() > self.c_W + self.d_W, 1, 0).reshape(-1)
        font = {
                'fontsize' : 8,
        }
        plt.title("name : " + name + "\npruning : " + str(w_pru.sum().data / len(w)) + "\nclipping : " + str(w_cli.sum().data / len(w)), fontdict = font, pad = 20)
        plt.subplot(3, 1, 2)
        plt.xlim(-np.max(np.abs(w)), np.max(np.abs(w)))
        plt.hist(w_quant, bins = 256)
        plt.axvline(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy()), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy()), color = "red")

        plt.subplot(3, 1, 3)
        plt.xlim(-np.max(np.abs(w)), np.max(np.abs(w)))
        plt.hist(w, bins =10000)
        plt.hist(w_quant, bins = 256)
        plt.axvline(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy(), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() - self.d_W.clone().cpu().detach().numpy()), color = "red")
        plt.axvline(-(self.c_W.clone().cpu().detach().numpy() + self.d_W.clone().cpu().detach().numpy()), color = "red")
        print(name, len(w), w_pru.sum().data / len(w), w_cli.sum().data / len(w))
        plt.savefig('./fig/' + name + '.png')
        plt.clf()

    def box_plot(self, name):
        w_abs = self.weight.abs().reshape(-1).sort()[0].clone().detach().cpu().numpy()
        plt.subplot(2, 1 ,1)
        plt.boxplot(w_abs)
        plt.hlines(w_abs[int(len(w_abs) * 0.7)], 0.75, 1.25)
        plt.hlines(w_abs[int(len(w_abs) * 0.9)], 0.75, 1.25)
        plt.hlines(w_abs[int(len(w_abs) * 0.95)], 0.75, 1.25)
        plt.title(name)
        if "query" in name or "value" in name or "key" in name:
            plt.subplot(2, 1, 2)
            w_abs = [self.weight[(i * 64): (i*64)+64, :].reshape(-1).clone().detach().cpu().numpy() for i in range(12)]
            plt.boxplot(w_abs)
        plt.savefig('./fig_box/' + name + '.png')
        plt.clf()
    def print_percentage(self, name):
        w = self.weight.clone().cpu().detach().reshape(-1).numpy()
        w_quant = (self.alpha_W * torch.abs(self.weight) + self.beta_W)
        w_quant = torch.clamp(w_quant, min = self.epsilon, max = 1.)
        w_quant = torch.where(w_quant > self.epsilon, w_quant ** self.gamma, w_quant * 0 )
        w_quant = self.weight_discretizer(w_quant, self.bitW) * torch.sign(self.weight)
        w_offset = self.c_W - self.d_W
        w_weight_scaling_factor = 2 * self.d_W
        w_quant = (torch.abs(w_quant) * w_weight_scaling_factor + w_offset) * torch.sign(w_quant)
        w_quant = w_quant.clone().cpu().detach().reshape(-1).numpy()
        w_pru = torch.where(self.weight.abs() < self.c_W - self.d_W, 1, 0).reshape(-1)
        w_cli = torch.where(self.weight.abs() > self.c_W + self.d_W, 1, 0).reshape(-1)
        print(name, len(w), w_pru.sum().data / len(w), w_cli.sum().data / len(w))
        return len(w), w_pru, w_cli
'''
def batch_frexp(inputs, max_bit = 31):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Args:
        Scaling_factor('torch.Tensor'):
            Target scaling factor to decompose

    Returns:
        ''Tuple(torch.Tensor, torch.Tensor)' : mantissa and exponent
    """

    shape_of_input = inputs.size()

    # trans the input to be a 1-d tensor
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().numpy())
    print(output_m, output_e)
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(
                decimal.Decimal(m * (2 ** max_bit)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP)
        )
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)
    output_e = float(max_bit) - output_e
    return (
            torch.from_numpy(output_m).to(inputs.device).view(shape_of_input),
            torch.from_numpy(output_e).to(inputs.device).view(shape_of_input),
    )
'''
'''
class FixedpointMul(Function):
    """
    Function to perform fixed-point arithmetic that can match integer arithmetic on HW.

    Args:
        pre_act ( 'torch.Tensor'):
            Input Tensor.
        pre_act_scaling_factor ( 'torch.Tensor'):
            Scaling factor of the input tensor *pre_act*
        bit_num ( 'int'):
            Quantization bitwidth.
        z_scaling_factor ( 'torch.Tensor' ):
            Scaling factor of the output tensor.
        identity ( 'torch.Tensor', *optional*):
            Identity tensor, if exists.
        identity_scaling_factor('torch.Tensor', *optional*):
            scaling factor of the identity tensor *identity*, if exists.
    Returns:
        'torch.Tensor':Output tensor(*pre_act* if *identity* is not given, otherwise the addition of *pre_act* and
        *identity*), whose scale is rescaled to *z_scaling_factor*.
    """
    @staticmethod
    def forward(
            ctx,
            pre_act,
            pre_act_scaling_factor,
            pre_act_offset,
            bit_num,
            z_scaling_factor,
            identity = None,
            identity_scaling_factor = None,
    ):
        ctx.identity = identity
        
        n = 2 ** (bit_num - 1) - 1

        with torch.no_grad():
            ctx.z_scaling_factor = z_scaling_factor

            z_int = torch.round(pre_act / pre_act_scaling_factor)
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B

            m, e = batch_frexp(new_scale)
            
            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / (2.0**e))

            if identity is not None:
                # needs addition of identity activation
                wx_int = torch.round(identity / identity_scaling_factor)

class testNet(nn.Module):
    def __init__(self, bitW = 2, bitA = 2,quant_mode = False):
        super(testNet, self).__init__()

        self.Embedding = QuantEmbedding(7, 10, weight_bit = bitW, quant_mode = quant_mode)
        self.linear = QILLinear(10, 2, weight_bit = bitW, quant_mode = quant_mode)
    def forward(self,x):
        x = self.Embedding(x)
        x = self.linear(x)
        return x
'''
if __name__ == "__main__":
    '''
    Linear_test = QILLinear(10, 8, weight_bit = 2,per_group = False, num_groups = 8,quant_mode = True)
    print(Linear_test.weight)
    print((Linear_test.weight.max(dim = -1)[0] - Linear_test.weight.min(dim = -1)[0]) /2)
    args = None
    Linear_test.quantize_reset_parameter(args)
    x = torch.rand((1, 10))
    print(x, "x")
    print(Linear_test(x))
    '''
    Act_test = QILQuantAct(bitA = 8, quant_mode = False, per_group = True, num_groups = 5)
    x = torch.rand((2, 10))
    print(x)
    print(Act_test(x))
    Act_test.act_init_mode = True
    y= Act_test(x)
    print(y)
    Act_test.quant_mode = True
    Act_test.act_init_mode = False
    y = Act_test(x)

