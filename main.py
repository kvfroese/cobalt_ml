import torch
import numpy
import configargparse

# Parsing

# Manual Operations

# Descriptors
'''
Create custom descriptor class for each descriptor (bond length, atomic number, etc.)
that inherits from torch.autograd.Function and implements forward and backward methods.
This allows us to compute the descriptor in the forward pass and its gradient in the backward pass
'''
class CustomDescriptor1(torch.autograd.Function): # to be called later by torch.nn.Module
    @staticmethod
    def forward(ctx, X): # number of inputs must match the number of gradients returned by backward
        desc_out = None # insert descriptor formula here
        # TODO need activation function
        ctx.save_for_backward(X) # need to save both input and intermediate descriptor for backward pass
        return desc_out

    @staticmethod
    def backward(ctx, grad_output):
        X, = ctx.saved_tensors
        dD_dX = None # insert derivative of descriptor with respect to input here
        grad_X = grad_output * dD_dX  # chain rule
        return grad_X

class CombinedDescriptors(torch.nn.Module):
    def forward(self, input):
        desc1 = CustomDescriptor1.apply(input)
        desc2 = None # update this
        combined_descriptor = torch.stack((desc1, desc2), dim=-1) # stack along last dimension
        return combined_descriptor