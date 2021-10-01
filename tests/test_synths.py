import numpy as np
import torch
from scipy.signal import find_peaks

from rt_ddsp import synths


def get_frequency_peaks(signal, sample_rate):
    spectrum = np.abs(np.fft.rfft(signal.numpy()))
    peaks, _ = find_peaks(spectrum, height=1)
    peak_freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)[peaks]
    # TODO: Why * 2?
    peak_amps = spectrum[peaks] / len(signal) * 2.

    return peak_freqs, peak_amps


def get_batch_frequency_peaks(signal, sample_rate):
    peak_freqs = []
    peak_amps = []
    for s in signal:
        pf, pa = get_frequency_peaks(s, sample_rate)
        peak_freqs.append(pf)
        peak_amps.append(pa)

    return np.stack(peak_freqs), np.stack(peak_amps)


def static_sawtooth_features(fundamental_frequency, base_amplitude, n_harmonics=30,
                             n_frames=1000, batch_size=3):
    amp = torch.zeros(batch_size, n_frames, 1) + base_amplitude

    harmonic_distribution = 1 / torch.arange(1, n_harmonics + 1)
    harmonic_distribution.view(1, 1, n_harmonics).repeat(batch_size, n_frames, 1)

    f0_hz = torch.zeros(batch_size, n_frames, 1) + fundamental_frequency

    return {
        'amplitudes': amp,
        'harmonic_distribution': harmonic_distribution,
        'f0_hz': f0_hz
    }


def test_harmonic_synth_is_accurate() -> None:
    sample_rate = 16000
    n_harmonics = 30
    n_frames = 1000
    batch_size = 3
    # f0 = np.array(batch_size * [220.0])
    # amp = np.array(batch_size * [0.6])
    f0 = 220.0
    amp = 0.6

    harmonic = synths.Harmonic(scale_fn=None)
    controls = static_sawtooth_features(f0, amp, n_harmonics, n_frames, batch_size)
    signal = harmonic(**controls)
    modified_controls = harmonic.get_controls(**controls)

    peak_freqs, peak_amps = get_batch_frequency_peaks(signal, sample_rate)

    expected_peak_freqs = np.stack(batch_size * [np.arange(1, n_harmonics + 1) * f0])

    expected_peak_amps = modified_controls['harmonic_distribution'][:, 0, :] * \
                         modified_controls['amplitudes'][:, 0, :]

    # filter above nyquist
    # expected_peak_freqs = expected_peak_freqs[expected_peak_freqs < sample_rate / 2]
    # expected_peak_amps = expected_peak_amps[expected_peak_freqs < sample_rate / 2]

    np.testing.assert_array_almost_equal(peak_freqs, expected_peak_freqs)
    np.testing.assert_array_almost_equal(peak_amps, expected_peak_amps)


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
