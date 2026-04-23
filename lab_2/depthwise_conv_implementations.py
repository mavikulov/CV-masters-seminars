import torch
import torch.nn.functional as F


def depthwise_conv2d(input_tensor: torch.Tensor, depthwise_filters: torch.Tensor, stride: int=1, padding: int=0) -> torch.Tensor:
    batch_size, C_in, H_in, W_in = input_tensor.shape
    kernel_size = depthwise_filters.shape[2]
    
    if padding > 0:
        input_padded = F.pad(input_tensor, (padding, padding, padding, padding))
    else:
        input_padded = input_tensor
    
    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1
    output = torch.zeros(batch_size, C_in, H_out, W_out)
    
    for b in range(batch_size):
        for c in range(C_in):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride
                    h_end = h_start + kernel_size
                    w_end = w_start + kernel_size
                    window = input_padded[b, c, h_start:h_end, w_start:w_end]
                    output[b, c, h, w] = (window * depthwise_filters[c, 0]).sum()
    return output


def pointwise_conv2d(input_tensor: torch.Tensor, pointwise_filters: torch.Tensor) -> torch.Tensor:
    batch_size, C_in, H_in, W_in = input_tensor.shape
    C_out = pointwise_filters.shape[0]
    output = torch.zeros(batch_size, C_out, H_in, W_in)
    
    for b in range(batch_size):
        for oc in range(C_out):
            for ic in range(C_in):
                output[b, oc] += input_tensor[b, ic] * pointwise_filters[oc, ic, 0, 0]
    
    return output


def depthwise_separable_conv2d(input_tensor: torch.Tensor, depthwise_filters: torch.Tensor, pointwise_filters: torch.Tensor, stride: int=1, padding: int=0) -> torch.Tensor:
    depthwise_output = depthwise_conv2d(input_tensor, depthwise_filters, stride, padding)
    output = pointwise_conv2d(depthwise_output, pointwise_filters)
    return output