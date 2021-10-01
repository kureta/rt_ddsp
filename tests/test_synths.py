from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch
from hypothesis import given, strategies as st
from scipy.signal import find_peaks  # type: ignore

from rt_ddsp import synths


def get_frequency_peaks(signal: torch.Tensor,
                        sample_rate: int,
                        height: float) -> Tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(np.fft.rfft(signal.numpy())) / (len(signal) / 2)
    peaks, _ = find_peaks(spectrum, height=height)
    peak_freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)[peaks]
    peak_amps = spectrum[peaks]

    return peak_freqs, peak_amps


def get_batch_frequency_peaks(signal: torch.Tensor,
                              sample_rate: int,
                              height: float) -> Tuple[np.ndarray, np.ndarray]:
    peak_freqs = []
    peak_amps = []
    for s in signal:
        pf, pa = get_frequency_peaks(s, sample_rate, height)
        peak_freqs.append(pf)
        peak_amps.append(pa)

    return np.stack(peak_freqs), np.stack(peak_amps)


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


@pytest.fixture(scope='module')
def harmonic_synth_16k() -> synths.Harmonic:
    return synths.Harmonic(16000 * 2, 16000)


@pytest.fixture(scope='module')
def harmonic_synth_44_1k() -> synths.Harmonic:
    return synths.Harmonic(44100 * 2, 44100)


# TODO: n_frames, n_samples, and seconds are tightly coupled

# Note: using half integers as f0 to hlp peak detection. Also 32-bit floats for amp.
@pytest.mark.parametrize(
    'harmonic_synth',
    ['harmonic_synth_16k', 'harmonic_synth_44_1k']
)
@given(
    n_harmonics=st.integers(1, 20),
    batch_size=st.integers(1, 8),
    double_f0=st.integers(200, 4000),
    amp=st.floats(1/8, 1.0, width=32)
)
def test_harmonic_synth_is_accurate(n_harmonics: int,
                                    batch_size: int,
                                    double_f0: int, amp: float,
                                    harmonic_synth: synths.Harmonic,
                                    request: Any) -> None:
    n_frames = 500
    f0 = double_f0 / 2.0

    harmonic_synth = request.getfixturevalue(harmonic_synth)
    sample_rate = harmonic_synth.sample_rate
    controls = static_sawtooth_features(f0, amp, n_harmonics, n_frames, batch_size)
    signal = harmonic_synth(**controls)
    modified_controls = harmonic_synth.get_controls(**controls)

    shit_mask = modified_controls['harmonic_distribution'][0, 0] > 0.
    height = (modified_controls['harmonic_distribution'][:, :, shit_mask] * modified_controls[
        'amplitudes']).numpy().min() * 0.95
    peak_freqs, peak_amps = get_batch_frequency_peaks(
        signal, sample_rate, height)

    expected_peak_freqs = np.stack(batch_size * [np.arange(1, n_harmonics + 1) * f0])

    dist_amps = modified_controls['harmonic_distribution'][:, 0, :]
    base_amps = modified_controls['amplitudes'][:, 0, :]
    expected_peak_amps = (dist_amps * base_amps).numpy()

    # filter above nyquist
    # TODO: currently we are assuming the whole batch has the same f0, harmonic, and amp values
    #       otherwise peak_freqs and expected_peak_frames would have different sizes
    #       later handle this by zero padding the smaller one or something like that
    mask = expected_peak_freqs[0] < sample_rate / 2
    expected_peak_amps = expected_peak_amps[:, mask]
    expected_peak_freqs = expected_peak_freqs[:, mask]

    np.testing.assert_array_almost_equal(peak_freqs, expected_peak_freqs, decimal=5)
    np.testing.assert_array_almost_equal(peak_amps, expected_peak_amps, decimal=5)


def test_harmonic_output_shape_is_correct() -> None:
    synthesizer = synths.Harmonic(
        n_samples=64000,
        sample_rate=16000,
        scale_fn=None,
        normalize_below_nyquist=True)
    batch_size = 3
    num_frames = 1000
    amp = torch.zeros((batch_size, num_frames, 1), dtype=torch.float32) + 1.0
    harmonic_distribution = torch.zeros(
        (batch_size, num_frames, 16), dtype=torch.float32) + 1.0 / 16
    f0_hz = torch.zeros((batch_size, num_frames, 1), dtype=torch.float32) + 16000

    output = synthesizer(amp, harmonic_distribution, f0_hz)

    assert [batch_size, 64000] == list(output.shape)


def test_filtered_noise_output_shape_is_correct() -> None:
    synthesizer = synths.FilteredNoise(n_samples=16000)
    filter_bank_magnitudes = torch.zeros((3, 16000, 100), dtype=torch.float32) + 3.0
    output = synthesizer(filter_bank_magnitudes)

    assert [3, 16000] == list(output.shape)
