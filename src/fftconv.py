import torch
import torch.nn.functional as F  # noqa
import torch.fft as fft


def fft_conv1d(signal, kernel):
    padded_kernel = F.pad(kernel, (0, signal.shape[-1] - kernel.shape[-1]))

    f_signal = fft.rfft(signal)
    f_kernel = fft.rfft(padded_kernel)

    f_kernel = torch.conj(f_kernel)

    f_conv = f_signal * f_kernel
    f_conv = fft.irfft(f_conv)
    f_conv = f_conv[..., :signal.shape[-1] - kernel.shape[-1] + 1]

    return f_conv


def grouped_fft_conv1d(signal, kernel):
    # merge batch, sequence, channel dimensions
    signal_ = signal.reshape(signal.shape[0] * signal.shape[1] * signal.shape[2],
                             signal.shape[3])
    kernel_ = kernel.reshape(kernel.shape[0] * kernel.shape[1] * kernel.shape[2],
                             kernel.shape[3])
    # add batch dim
    signal_ = signal_.unsqueeze(0)
    # add out channel dim
    kernel_ = kernel_.unsqueeze(0)

    f_conv = fft_conv1d(signal_, kernel_)
    f_conv = f_conv.reshape(signal.shape[0], signal.shape[1], signal.shape[2], -1)

    return f_conv
