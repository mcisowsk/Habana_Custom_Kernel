import torch
import os
import habana_frameworks.torch.core

custom_reinterpret_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_reinterpret.cpython-310-x86_64-linux-gnu.so" # XXX this must match exactly! Check if this file exists befoe proceeding
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_reinterpret_op_lib_path))

class CustomReinterpretFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_reinterpret(value)
        ctx.tensor = tensor
        return tensor

class CustomReinterpret(torch.nn.Module):
    def __init__(self):
        super(CustomReinterpret, self).__init__()

    def forward(self, value):
        return CustomReinterpretFunction.apply(value)

    def extra_repr(self):
        return 'CustomReinterpret for float32 only' # XXX ???

