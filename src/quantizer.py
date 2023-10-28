import torch
import torch.nn as nn

def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)


def block_quantize(data,bits,mode,ebits):
    max_exponent = torch.floor(torch.log2(torch.abs(torch.where(
            data==torch.zeros_like(data),torch.ones_like(data),data ))))
    max_exponent.clamp_(-2**(ebits-1),2**(ebits-1)-1)
    i = data * 2 ** (-max_exponent+(bits-2))
    if mode == 'stochastic':
        add_r_(i)
        i.floor_()
    elif mode == 'nearest':
        i.round_()
    i.clamp_(-2**(bits-1),2**(bits-1)-1)
    temp = i * 2 ** (max_exponent-(bits-2))
    return temp

def q_quantize(data,bits,mode,ebits):
    max_exponent = torch.floor(torch.log2(torch.abs(torch.where(
        data == torch.zeros_like(data), torch.ones_like(data), data))))
    max_exponent.clamp_(-2 ** (ebits - 1), 2 ** (ebits - 1) - 1)
    i = data * 2 ** (-max_exponent + (bits - 2))
    cur_exp = 2 ** (max_exponent - (bits - 2))
    p4left = 1 - i % 1
    p4right = i % 1
    q_n_left = torch.floor(i).clamp_(-2 ** (bits - 1),2**(bits - 1) - 1) * cur_exp
    q_n_right = torch.ceil(i).clamp_(-2 ** (bits - 1),2**(bits - 1) - 1) * cur_exp
    e_q = torch.pow(q_n_left - data , 2) * p4left + torch.pow(q_n_right - data,2) * p4right
    powdata = torch.pow(data,2)
    q = e_q / powdata
    end_q = torch.where(data==torch.zeros_like(data),torch.zeros_like(data),q)
    if mode == 'stochastic':
        add_r_(i)
        i.floor_()
    elif mode == 'nearest':
        i.round_()
    i.clamp_(-2**(bits-1),2**(bits-1)-1)
    temp = i * 2**(max_exponent-(bits-2))
    max_q = torch.max(end_q)
    ind = torch.where(torch.abs(end_q-max()>=1e-9),-1000*torch.ones_like(end_q),data)
    data_q = torch.max(ind)
    return temp,max_q,torch.max(data_q)

class BlockRounding(torch.autograd.Function):
    @staticmethod
    def forward(self,x,bits,ebits,mode):
        self.ebits = bits
        self.bits = bits
        self.mode = mode
        if bits == -1 :
            return x
        return block_quantize(x,bits,self.mode,ebits)

    @staticmethod
    def backward(self,grad_output):
        if self.needs_input_grad[0]:
            if self.bits != -1:
                grad_input = block_quantize(grad_output,self.bits,self.mode,self.ebits)
            else:
                grad_input = grad_output
        return grad_input,None,None,None,None


quantize_block = BlockRounding.apply # 加载量化

class BlockQuantizer(nn.Module):
    def __init__(self,bits,ebits,mode):
        super(BlockQuantizer,self).__init__()
        self.bits = bits
        self.ebits = ebits
        self.mode = mode

    def forward(self,x):
        return quantize_block(x,self.bits,self.ebits,self.mode)