from typing import Dict, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks  # type: ignore


# TODO: n_frames, n_samples, and seconds are tightly coupled, at least one is redundant.
def get_frequency_peaks(signal: torch.Tensor,
                        sample_rate: int,
                        height: float) -> Tuple[torch.Tensor, torch.Tensor]:
    spectrum = np.abs(np.fft.rfft(signal.numpy())) / (len(signal) / 2)
    peaks, _ = find_peaks(spectrum, height=height)
    peak_freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)[peaks]
    peak_amps = spectrum[peaks]

    return torch.tensor(peak_freqs, dtype=torch.float32), torch.tensor(peak_amps,
                                                                       dtype=torch.float32)


def get_batch_frequency_peaks(signal: torch.Tensor,
                              sample_rate: int,
                              height: float) -> Tuple[torch.Tensor, torch.Tensor]:
    peak_freqs = []
    peak_amps = []
    for s in signal:
        pf, pa = get_frequency_peaks(s, sample_rate, height)
        peak_freqs.append(pf)
        peak_amps.append(pa)

    return torch.stack(peak_freqs), torch.stack(peak_amps)


def static_sawtooth_features(fundamental_frequency: float,
                             base_amplitude: float,
                             n_harmonics: int = 30,
                             n_frames: int = 1000,
                             batch_size: int = 3) -> Dict[str, torch.Tensor]:
    amp = torch.zeros(batch_size, n_frames, 1) + base_amplitude

    harmonic_distribution = 1 / torch.arange(1, n_harmonics + 1)
    # harmonic_distribution = torch.ones(n_harmonics)  # impulse features
    harmonic_distribution.view(1, 1, n_harmonics).repeat(batch_size, n_frames, 1)

    f0_hz = torch.zeros(batch_size, n_frames, 1) + fundamental_frequency

    return {
        'amplitudes': amp,
        'harmonic_distribution': harmonic_distribution,
        'f0_hz': f0_hz
    }
