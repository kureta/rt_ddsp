from typing import Optional

import torch
import torch.nn as nn

from rt_ddsp import core, processors
from rt_ddsp.core import torch_float32
from rt_ddsp.types import TensorDict


class Reverb(processors.Processor):
    """Convolutional (FIR) reverb."""

    def __init__(self,
                 reverb_length: int = 48000,
                 add_dry: bool = True):
        """Takes neural network outputs directly as the impulse response.

        Args:
          trainable: Learn the impulse_response as a single variable for the
            entire dataset.
          reverb_length: Length of the impulse response. Only used if
            trainable=True.
          add_dry: Add dry signal to reverberated signal on output.
        """
        super().__init__()
        self._reverb_length = reverb_length
        self._add_dry = add_dry
        self._ir = nn.Parameter(
            torch.randn(1, 1, self._reverb_length) * 1e-6, requires_grad=True
        )

    @staticmethod
    def _mask_dry_ir(ir: torch.Tensor) -> torch.Tensor:
        """Set first impulse response to zero to mask the dry signal."""
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
        """Tile the impulse response variable to match the batch size."""
        # Add batch dimension.
        if len(ir.shape) == 1:
            ir = ir[None, :]
        # Match batch dimension.
        batch_size = int(audio.shape[0])
        return torch.tile(ir, [batch_size, 1])

    def get_controls(self, audio: torch.Tensor,  # type: ignore[override]
                     ir: Optional[torch.Tensor] = None) -> TensorDict:
        """Convert decoder outputs into ir response.

        Args:
          audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
          ir: 3-D Tensor of shape [batch, ir_size, 1] or 2D Tensor of shape
            [batch, ir_size].

        Returns:
          controls: Dictionary of effect controls.

        Raises:
          ValueError: If trainable=False and ir is not provided.
        """
        ir = self._match_dimensions(audio, self._ir)

        return {'audio': audio, 'ir': ir}

    def get_signal(self, audio: torch.Tensor,  # type: ignore[override]
                   ir: torch.Tensor) -> torch.Tensor:
        """Apply impulse response.

        Args:
          audio: Dry audio, 2-D Tensor of shape [batch, n_samples].
          ir: 3-D Tensor of shape [batch, ir_size, 1] or 2D Tensor of shape
            [batch, ir_size].

        Returns:
          tensor of shape [batch, n_samples]
        """
        audio, ir = torch_float32(audio), torch_float32(ir)
        ir = self._mask_dry_ir(ir)
        wet = core.fft_convolve(audio, ir, padding='same',
                                delay_compensation=0)
        return (wet + audio) if self._add_dry else wet
