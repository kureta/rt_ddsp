import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch import Tensor

from fftconv import fft_conv


class Reverb(nn.Module):
    def __init__(self,
                 sample_rate: int = 16000,
                 duration: float = 1.0,
                 batch_size: int = 1,
                 live: bool = False,
                 n_channels: int = 1):
        super().__init__()

        self.duration = int(sample_rate * duration)
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.live = live
        self.n_channels = n_channels

        # ir.shape = (out_channels, in_channels, size)
        self.ir = nn.Parameter(torch.rand(n_channels, n_channels, self.duration) * 2.0 - 1.0,
                               requires_grad=True)

        if self.live:
            self.buffer: Tensor
            self.register_buffer('buffer',
                                 torch.zeros(self.batch_size, n_channels, self.duration),
                                 persistent=False)

    def forward(self, signal: Tensor) -> Tensor:
        ir = self.ir.flip(-1)
        signal_length = signal.shape[-1]

        result = fft_conv(F.pad(signal, (self.duration - 1, self.duration)), ir)

        if self.live:
            # Separate reverberated signal and tail
            out = result[..., :signal_length]
            tail = result[..., signal_length:]

            # add AT MOST first signal_length samples of the old buffer to the result
            # reverb duration might be shorter than signal length.
            # In that case, tail of the previous signal
            # is shorter than the current signal.
            previous_tail = self.buffer[..., :signal_length]
            prev_tail_len = previous_tail.shape[-1]
            out[..., :prev_tail_len] += previous_tail

            # zero out used samples of the old buffer
            self.buffer[..., :prev_tail_len] = 0.0

            # roll used samples to the end
            self.buffer = self.buffer.roll(-prev_tail_len, dims=-1)  # noqa

            # add new tail to buffer
            self.buffer += tail

            return out
        else:
            return result[..., :signal_length]
