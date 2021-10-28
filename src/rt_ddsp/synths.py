from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch  # noqa
import torch.nn as nn  # noqa
import torch.nn.functional as F  # noqa

from rt_ddsp import core, processors
from rt_ddsp.processors import TensorDict


class OscillatorBank(nn.Module):
    def __init__(self, batch_size: int = 4, sample_rate: int = 16000,
                 n_harmonics: int = 100, hop_size: int = 512):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop_size = hop_size

        self.harmonics = nn.Parameter(
            torch.arange(1, self.n_harmonics + 1, step=1), requires_grad=False
        )
        self.last_phases = nn.Parameter(
            # torch.rand(batch_size, n_harmonics) * 2. * np.pi - np.pi, requires_grad=False
            torch.zeros(batch_size, n_harmonics), requires_grad=False
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

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        f0 = x['f0_hz']
        harm_amps = x['harmonic_distribution']
        loudness = x['amplitudes']

        harmonics, harm_amps = self.prepare_harmonics(f0, harm_amps)
        harmonics[:, 0, :] += self.last_phases  # phase offset from last sample
        phases = self.generate_phases(harmonics)
        self.last_phases[:] = phases[:, -1, :]  # update phase offset
        signal = self.generate_signal(harm_amps, loudness, phases)

        return signal


class Harmonic(processors.Processor):
    # TODO: instead of n_samples this should use something like control rate and
    #       deduce n_samples from there.
    def __init__(self,
                 n_samples: int = 64000,
                 sample_rate: int = 16000,
                 scale_fn: Optional[Callable] = core.exp_sigmoid,
                 normalize_below_nyquist: bool = True,
                 amp_resample_method: str = 'linear'):
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.scale_fn = scale_fn
        self.normalize_below_nyquist = normalize_below_nyquist
        self.amp_resample_method = amp_resample_method

    def get_controls(self,  # type: ignore[override]
                     amplitudes: torch.Tensor,
                     harmonic_distribution: torch.Tensor,
                     f0_hz: torch.Tensor) -> TensorDict:
        # Scale the amplitudes.
        if self.scale_fn is not None:
            amplitudes = self.scale_fn(amplitudes)
            harmonic_distribution = self.scale_fn(harmonic_distribution)

        # Bandlimit the harmonic distribution.
        if self.normalize_below_nyquist:
            n_harmonics = int(harmonic_distribution.shape[-1])
            harmonic_frequencies = core.get_harmonic_frequencies(f0_hz,
                                                                 n_harmonics)
            harmonic_distribution = core.remove_above_nyquist(
                harmonic_frequencies,
                harmonic_distribution,
                self.sample_rate)

        # Normalize
        harmonic_distribution /= torch.sum(harmonic_distribution,
                                           dim=-1,
                                           keepdim=True)

        return {'amplitudes': amplitudes,
                'harmonic_distribution': harmonic_distribution,
                'f0_hz': f0_hz}

    def get_signal(self,  # type: ignore[override]
                   amplitudes: torch.Tensor,
                   harmonic_distribution: torch.Tensor,
                   f0_hz: torch.Tensor) -> torch.Tensor:
        signal = core.harmonic_synthesis(
            frequencies=f0_hz,
            amplitudes=amplitudes,
            harmonic_distribution=harmonic_distribution,
            n_samples=self.n_samples,
            sample_rate=self.sample_rate,
            amp_resample_method=self.amp_resample_method)
        return signal


class FilteredNoise(processors.Processor):
    def __init__(self,
                 n_samples: int = 64000,
                 window_size: int = 257,
                 scale_fn: Optional[Callable] = core.exp_sigmoid,
                 initial_bias: float = -5.0):
        super().__init__()
        self.n_samples = n_samples
        self.window_size = window_size
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias

    def get_controls(self, magnitudes: torch.Tensor) -> TensorDict:  # type: ignore[override]
        # Scale the magnitudes.
        if self.scale_fn is not None:
            magnitudes = self.scale_fn(magnitudes + self.initial_bias)

        return {'magnitudes': magnitudes}

    def get_signal(self, magnitudes: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch_size = int(magnitudes.shape[0])
        signal = torch.rand([batch_size, self.n_samples]) * 2.0 - 1.0
        return core.frequency_filter(signal,
                                     magnitudes,
                                     window_size=self.window_size)
