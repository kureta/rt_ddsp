from typing import Optional

import torch
import torch.nn as nn

from rt_ddsp import core, processors
from rt_ddsp.core import torch_float32
from rt_ddsp.types import TensorDict


class Reverb(processors.Processor):
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
