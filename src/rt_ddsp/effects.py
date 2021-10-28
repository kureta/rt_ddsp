from typing import Optional

import torch
import torch.nn as nn

from rt_ddsp import core, processors
from rt_ddsp.core import torch_float32
from rt_ddsp.types import TensorDict

from fftconv import fft_conv


class Reverb(nn.Module):
    def __init__(self, sample_rate=16000, duration=1.0, batch_size=1, live=False):
        super().__init__()

        self.duration = int(sample_rate * duration)
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.live = live

        self.ir = nn.Parameter(torch.rand(self.duration) * 2.0 - 1.0, requires_grad=True)
        self.register_buffer('buffer', torch.zeros(self.batch_size, 1, self.duration),
                             persistent=False)

    def forward(self, signal):
        if self.live:
            with torch.no_grad():
                return self.forward_live(signal)
        else:
            return self.forward_learn(signal)

    def forward_learn(self, signal):
        ir = self.ir[None, None, :].flip(-1)
        signal_length = signal.shape[-1]

        result = fft_conv(signal, ir, padding=self.duration)

        return result[..., :signal_length]

    def forward_live(self, signal):
        ir = self.ir[None, None, :].flip(-1)
        signal_length = signal.shape[-1]

        # TODO: Understand why this is so.
        # Drop the last residual sample
        result = fft_conv(signal, ir, padding=ir.shape[-1])[..., :-1]

        # Separate reverberated signal and tail
        out = result[..., :signal_length]
        tail = result[..., signal_length:]

        # add AT MOST first signal_length samples of the old buffer to the result
        # reverb duration might be shorter than signal length. In that case, tail of the
        # previous signal is shorter than the current signal.
        previous_tail = self.buffer[..., :signal_length]
        prev_tail_len = previous_tail.shape[-1]
        out[..., :prev_tail_len] += previous_tail

        # zero out used samples of the old buffer
        self.buffer[..., :prev_tail_len] = 0.0

        # roll used samples to the end
        self.buffer = self.buffer.roll(-prev_tail_len, dims=-1)

        # add new tail to buffer
        self.buffer += tail

        return out


class ReverbOld(processors.Processor):
    def __init__(self,
                 reverb_length: int = 48000,
                 add_dry: bool = True):
        super().__init__()
        self._reverb_length = reverb_length
        self._add_dry = add_dry
        self._ir = nn.Parameter(
            torch.randn(1, 1, self._reverb_length) * 1e-6, requires_grad=True
        )

    @staticmethod
    def _mask_dry_ir(ir: torch.Tensor) -> torch.Tensor:
        # Make IR 2-D [batch, ir_size].
        if len(ir.shape) == 1:
            ir = ir[None, :]  # Add a batch dimension
        if len(ir.shape) == 3:
            ir = ir[:, :, 0]  # Remove unnecessary channel dimension.
        # Mask the dry signal.
        dry_mask = torch.zeros(int(ir.shape[0]), 1)
        return torch.cat([dry_mask, ir[:, 1:]], dim=1)

    @staticmethod
    def _match_dimensions(audio: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        # Add batch dimension.
        if len(ir.shape) == 1:
            ir = ir[None, :]
        # Match batch dimension.
        batch_size = int(audio.shape[0])
        return torch.tile(ir, [batch_size, 1])

    def get_controls(self, audio: torch.Tensor,  # type: ignore[override]
                     ir: Optional[torch.Tensor] = None) -> TensorDict:
        ir = self._match_dimensions(audio, self._ir)

        return {'audio': audio, 'ir': ir}

    def get_signal(self, audio: torch.Tensor,  # type: ignore[override]
                   ir: torch.Tensor) -> torch.Tensor:
        audio, ir = torch_float32(audio), torch_float32(ir)
        ir = self._mask_dry_ir(ir)
        wet = core.fft_convolve(audio, ir, padding='same',
                                delay_compensation=0)
        return (wet + audio) if self._add_dry else wet
