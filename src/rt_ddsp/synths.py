from typing import Callable, Optional

import torch

from rt_ddsp import core, processors
from rt_ddsp.processors import TensorDict


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
