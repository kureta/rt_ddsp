from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from fftconv import fft_conv


class OscillatorBank(nn.Module):
    def __init__(self, batch_size: int = 4, sample_rate: int = 16000,
                 n_harmonics: int = 100, hop_size: int = 512, live: bool = False):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.live = live

        self.harmonics: torch.Tensor
        self.register_buffer(
            'harmonics',
            torch.arange(1, self.n_harmonics + 1, step=1), persistent=False
        )

        self.last_phases: torch.Tensor
        self.register_buffer(
            'last_phases',
            # torch.rand(batch_size, n_harmonics) * 2. * np.pi - np.pi, requires_grad=False
            torch.zeros(batch_size, n_harmonics), persistent=False
        )

    def prepare_harmonics(self, f0: torch.Tensor,
                          harm_amps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cut above Nyquist and normalize
        # Hz (cycles per second)
        harmonics = (
            self.harmonics.unsqueeze(0).unsqueeze(0).repeat(f0.shape[0], f0.shape[1], 1)
            * f0
        )
        # zero out above nyquist
        mask = harmonics > self.sample_rate // 2
        harm_amps = harm_amps.masked_fill(mask, 0.0)
        harm_amps /= harm_amps.sum(-1, keepdim=True)
        harmonics *= 2 * np.pi  # radians per second
        harmonics /= self.sample_rate  # radians per sample
        harmonics = self.rescale(harmonics)
        return harmonics, harm_amps

    @staticmethod
    def generate_phases(harmonics: torch.Tensor) -> torch.Tensor:
        phases = torch.cumsum(harmonics, dim=1)
        phases %= 2 * np.pi
        return phases

    def generate_signal(
        self, harm_amps: torch.Tensor, loudness: torch.Tensor, phases: torch.Tensor
    ) -> torch.Tensor:
        loudness = self.rescale(loudness)
        harm_amps = self.rescale(harm_amps)
        signal = loudness * harm_amps * torch.sin(phases)
        signal = torch.sum(signal, dim=2)
        return signal

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x.permute(0, 2, 1),
            scale_factor=float(self.hop_size),
            mode='linear',
            align_corners=True,
        ).permute(0, 2, 1)

    def forward(self,
                f0: torch.Tensor,
                loudness: torch.Tensor,
                harm_amps: torch.Tensor) -> torch.Tensor:
        harmonics, harm_amps = self.prepare_harmonics(f0, harm_amps)
        if self.live:
            harmonics[:, 0, :] += self.last_phases  # phase offset from last sample
        phases = self.generate_phases(harmonics)
        if self.live:
            self.last_phases[:] = phases[:, -1, :]  # update phase offset
        signal = self.generate_signal(harm_amps, loudness, phases)

        return signal


# TODO: lot's of duplicated code
class Noise(nn.Module):
    def __init__(self, sample_rate: int = 16000,
                 hop_size: int = 512,
                 batch_size: int = 1,
                 live: bool = False,
                 n_channels: int = 1):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.batch_size = batch_size
        self.live = live
        self.n_channels = n_channels

        self.unfold = nn.Unfold(kernel_size=(1, hop_size * 2), stride=(1, hop_size),
                                padding=(0, 0))

        self.buffer: torch.Tensor
        self.register_buffer('buffer',
                             torch.zeros(self.batch_size, n_channels, 2 * self.hop_size),
                             persistent=False)

        self.window: torch.Tensor
        self.register_buffer('window', torch.hann_window(hop_size * 2), persistent=False)

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        nir = torch.fft.irfft(bands, dim=-1)
        nir = torch.fft.fftshift(nir, dim=-1)

        if self.live:
            batch_size = self.batch_size
        else:
            batch_size = bands.shape[0]
        seq_len = bands.shape[1]

        noise = torch.rand(batch_size, 1, 1,
                           seq_len * self.hop_size + self.hop_size) * 2.0 - 1.0
        framed_noise = self.unfold(noise).permute(0, 2, 1)
        filtered = fft_conv(F.pad(framed_noise.reshape(1, -1, self.hop_size * 2),
                                  (self.sample_rate - 1, self.sample_rate)),
                            nir.reshape(batch_size * seq_len, 1, self.sample_rate),
                            groups=batch_size * seq_len)
        filtered = filtered[..., self.sample_rate // 2:-self.sample_rate // 2]
        windowed = filtered * self.window

        windowed = windowed.reshape(batch_size, seq_len, 2 * self.hop_size)
        windowed = windowed.permute(0, 2, 1)

        if self.live:
            windowed, self.buffer[:, 0, :] = torch.cat([self.buffer.permute(0, 2, 1), windowed],
                                                       dim=-1), windowed[:, :, -1]
        else:
            windowed = torch.cat(
                [torch.zeros(batch_size, 2 * self.hop_size, self.n_channels), windowed],
                dim=-1
            )

        result = F.fold(windowed, (1, self.hop_size * seq_len + 2 * self.hop_size),
                        kernel_size=(1, self.hop_size * 2), stride=(1, self.hop_size),
                        padding=(0, 0))
        result.squeeze_(1)

        return result[..., self.hop_size:-self.hop_size]
