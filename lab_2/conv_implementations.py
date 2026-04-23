import numpy as np


def naive_conv2d(input_: np.ndarray, kernel: np.ndarray, padding: int=0, stride: int=1) -> np.ndarray:
    if input_.ndim == 2:
        input_ = np.expand_dims(input_, axis=0)
    if kernel.ndim == 2:
        kernel = np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)
    
    C_in, H_in, W_in = input_.shape
    C_out, _, KH, KW = kernel.shape
    
    if padding > 0:
        padded = np.zeros((C_in, H_in + 2 * padding, W_in + 2 * padding))
        padded[:, padding:-padding, padding:-padding] = input_
    else:
        padded = input_
    
    H_out = (H_in + 2 * padding - KH) // stride + 1
    W_out = (W_in + 2 * padding - KW) // stride + 1
    output = np.zeros((C_out, H_out, W_out))
    
    for oc in range(C_out):
        for ic in range(C_in):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride

                    output[oc, h, w] += np.sum(
                        padded[ic, h_start:h_start + KH, w_start:w_start + KW] * kernel[oc, ic]
                    )
    
    return output.squeeze()


def im2col(input_: np.ndarray, kernel: np.ndarray, padding: int=0, stride: int=1) -> np.ndarray:
    if input_.ndim == 2:
        input_ = np.expand_dims(input_, axis=0)
    
    C_in, H_in, W_in = input_.shape
    KH, KW = kernel
    
    if padding > 0:
        padded_input = np.zeros((C_in, H_in + 2 * padding, W_in + 2 * padding))
        padded_input[:, padding:-padding, padding:-padding] = input_
    else:
        padded_input = input_
    
    H_out = (H_in + 2 * padding - KH) // stride + 1
    W_out = (W_in + 2 * padding - KW) // stride + 1
    col_matrix = np.zeros((C_in * KH * KW, H_out * W_out))
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            w_start = w * stride
            patch = padded_input[:, h_start:h_start + KH, w_start:w_start + KW]
            col_matrix[:, h * W_out + w] = patch.ravel()
    
    return col_matrix


def conv2d_im2col(input_: np.ndarray, kernel: np.ndarray, padding: int=0, stride: int=1) -> np.ndarray:
    if input_.ndim == 2:
        input_ = np.expand_dims(input_, axis=0)
    if kernel.ndim == 2:
        kernel = np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)
    
    C_in, H_in, W_in = input_.shape
    C_out, in_c_k, KH, KW = kernel.shape
    
    if C_in != in_c_k:
        raise ValueError(f"Количество каналов во входе ({C_in}) не совпадает с ядром ({in_c_k})")
    
    col_matrix = im2col(input_, (KH, KW), padding, stride)
    kernel_matrix = kernel.reshape(C_out, -1)
    out = kernel_matrix @ col_matrix
    H_out = (H_in + 2 * padding - KH) // stride + 1
    W_out = (W_in + 2 * padding - KW) // stride + 1
    
    return out.reshape(C_out, H_out, W_out).squeeze()
