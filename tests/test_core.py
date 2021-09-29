from typing import Tuple

import numpy as np
import pytest
import torch
from scipy import signal

from rt_ddsp import core
from rt_ddsp.core import torch_float32


# TODO: Add hypothesis tests
def create_wave_np(batch_size: int, frequencies: torch.Tensor, amplitudes: torch.Tensor,
                   seconds: float, n_samples: int) -> np.ndarray:
    wave_np = np.zeros([batch_size, n_samples])
    time = np.linspace(0, seconds, n_samples)
    n_harmonics = int(frequencies.shape[-1])
    for i in range(batch_size):
        for j in range(n_harmonics):
            rads_per_cycle = 2.0 * np.pi
            rads_per_sec = rads_per_cycle * frequencies[i, :, j]
            phase = time * rads_per_sec
            wave_np[i, :] += amplitudes[i, :, j] * np.sin(phase)
    return wave_np


HarmonicSynthSettings = Tuple[int, int, float, int]


@pytest.fixture
def harmonic_synth_settings() -> HarmonicSynthSettings:
    batch_size = 2
    sample_rate = 16000
    seconds = 1.0
    n_samples = int(seconds) * sample_rate

    return batch_size, sample_rate, seconds, n_samples


@pytest.mark.parametrize(
    'batch_size, fundamental_frequency, n_harmonics, sample_rate, seconds',
    [
        (2, 62.4, 5, 16000, 2),
        (16, 100, 1, 8000, 0.5),
        (1, 2000, 2, 4000, 1.3),
    ]
)
def test_oscillator_bank_is_accurate(batch_size: int, fundamental_frequency: float,
                                     n_harmonics: int, sample_rate: int,
                                     seconds: float) -> None:
    n_samples = int(sample_rate * seconds)
    seconds = float(n_samples) / sample_rate
    frequencies = fundamental_frequency * np.arange(1, n_harmonics + 1)
    amplitudes = 1.0 / n_harmonics * np.ones_like(frequencies)

    # Create tensors of frequencies and amplitudes for tf function.
    ones = np.ones([batch_size, n_samples, n_harmonics])
    frequency_envelopes = ones * frequencies[np.newaxis, np.newaxis, :]
    amplitude_envelopes = ones * amplitudes[np.newaxis, np.newaxis, :]

    # Create np test signal.
    wav_np = create_wave_np(batch_size, frequency_envelopes,
                            amplitude_envelopes, seconds, n_samples)

    wav_tf = core.oscillator_bank(
        torch_float32(frequency_envelopes),
        torch_float32(amplitude_envelopes), sample_rate=sample_rate)

    pad = 10  # Ignore edge effects.
    np.testing.assert_array_almost_equal(wav_np[pad:-pad], wav_tf[pad:-pad].numpy())


@pytest.mark.parametrize(
    'sum_sinusoids',
    [
        True, False
    ]
)
def test_oscillator_bank_shape_is_correct(
        sum_sinusoids: bool,
        harmonic_synth_settings: HarmonicSynthSettings) -> None:
    batch_size, sample_rate, seconds, n_samples = harmonic_synth_settings
    frequencies = np.array([1.0, 1.5, 2.0]) * 400.0
    amplitudes = np.ones_like(frequencies)

    # Create tensors of frequencies and amplitudes for tf function.
    ones = torch.ones([batch_size, n_samples, 3])
    frequency_envelopes = ones * frequencies[np.newaxis, np.newaxis, :]
    amplitude_envelopes = ones * amplitudes[np.newaxis, np.newaxis, :]

    wav_tf = core.oscillator_bank(frequency_envelopes,
                                  amplitude_envelopes,
                                  sample_rate=sample_rate,
                                  sum_sinusoids=sum_sinusoids)
    if sum_sinusoids:
        expected_shape = [batch_size, n_samples]
    else:
        expected_shape = [batch_size, n_samples, 3]
    assert expected_shape == list(wav_tf.shape)


@pytest.mark.parametrize(
    'sample_rate',
    [
        4000, 16000, 44100
    ]
)
def test_silent_above_nyquist(sample_rate: int,
                              harmonic_synth_settings: HarmonicSynthSettings) -> None:
    batch_size, sample_rate, seconds, n_samples = harmonic_synth_settings
    nyquist = sample_rate / 2
    frequencies = torch.tensor([1.1, 1.5, 2.0]) * nyquist
    amplitudes = torch.ones_like(frequencies)

    # Create tensors of frequencies and amplitudes for tf function.
    ones = torch.ones([batch_size, n_samples, 3])
    frequency_envelopes = ones * frequencies[None, None, :]
    amplitude_envelopes = ones * amplitudes[None, None, :]

    wav_tf = core.oscillator_bank(
        frequency_envelopes, amplitude_envelopes, sample_rate=sample_rate)
    wav_np = np.zeros_like(wav_tf.numpy())
    np.testing.assert_array_almost_equal(wav_np, wav_tf.numpy())


@pytest.mark.parametrize(
    'batch_size, fundamental_frequency, amplitude, n_frames',
    [
        (2, 20, 0.1, 100),
        (1, 100, 0.2, 1000),
        (4, 2000, 0.5, 100),
    ]
)
def test_harmonic_synthesis_is_accurate_one_frequency(
        batch_size: int,
        fundamental_frequency: float,
        amplitude: float,
        n_frames: int,
        harmonic_synth_settings: HarmonicSynthSettings) -> None:
    batch_size, sample_rate, seconds, n_samples = harmonic_synth_settings
    frequencies = fundamental_frequency * np.ones([batch_size, n_frames, 1])
    amplitudes = amplitude * np.ones([batch_size, n_frames, 1])

    frequencies_np = fundamental_frequency * np.ones(
        [batch_size, n_samples, 1])
    amplitudes_np = amplitude * np.ones([batch_size, n_samples, 1])

    # Create np test signal.
    wav_np = create_wave_np(batch_size, frequencies_np, amplitudes_np,
                            seconds, n_samples)

    wav_tf = core.harmonic_synthesis(
        torch_float32(frequencies),
        torch_float32(amplitudes),
        n_samples=n_samples,
        sample_rate=sample_rate)
    pad = n_samples // n_frames  # Ignore edge effects.
    np.testing.assert_array_almost_equal(wav_np[pad:-pad], wav_tf[pad:-pad].numpy())


@pytest.mark.parametrize(
    'n_harmonics',
    [
        1, 20, 40
    ]
)
def test_harmonic_synthesis_is_accurate_multiple_harmonics(
        n_harmonics: int,
        harmonic_synth_settings: HarmonicSynthSettings) -> None:
    batch_size, sample_rate, seconds, n_samples = harmonic_synth_settings
    fundamental_frequency = 440.0
    amp = 0.1
    n_frames = 100

    harmonic_shifts = np.abs(np.random.randn(1, 1, n_harmonics))
    harmonic_distribution = np.abs(np.random.randn(1, 1, n_harmonics))

    frequencies_tf = fundamental_frequency * np.ones(
        [batch_size, n_frames, 1])
    amplitudes_tf = amp * np.ones([batch_size, n_frames, 1])
    harmonic_shifts_tf = np.tile(harmonic_shifts, [1, n_frames, 1])
    harmonic_distribution_tf = np.tile(harmonic_distribution, [1, n_frames, 1])

    # Create np test signal.
    frequencies_np = fundamental_frequency * np.ones(
        [batch_size, n_samples, 1])
    amplitudes_np = amp * np.ones([batch_size, n_samples, 1])
    frequencies_np = frequencies_np * harmonic_shifts
    amplitudes_np = amplitudes_np * harmonic_distribution
    wav_np = create_wave_np(batch_size, frequencies_np, amplitudes_np,
                            seconds, n_samples)

    wav_tf = core.harmonic_synthesis(
        torch.from_numpy(frequencies_tf),
        torch.from_numpy(amplitudes_tf),
        torch.from_numpy(harmonic_shifts_tf),
        torch.from_numpy(harmonic_distribution_tf),
        n_samples=n_samples,
        sample_rate=sample_rate)
    pad = n_samples // n_frames  # Ignore edge effects.
    np.testing.assert_array_almost_equal(wav_np[pad:-pad], wav_tf[pad:-pad].numpy())


@pytest.fixture
def audio() -> np.ndarray:
    audio_size = 1000
    return np.random.randn(1, audio_size).astype(np.float32)


@pytest.mark.parametrize(
    'audio_size, impulse_response_size',
    [
        (1000, 10),
        (10, 100)
    ]
)
def test_fft_convolve_is_accurate(audio_size: int, impulse_response_size: int) -> None:
    # Create random signals to convolve.
    audio = np.ones([1, audio_size]).astype(np.float32)
    impulse_response = np.ones([1, impulse_response_size]).astype(np.float32)

    output_tf = core.fft_convolve(
        torch.from_numpy(audio), torch.from_numpy(impulse_response),
        padding='valid', delay_compensation=0)[0]

    output_np = signal.fftconvolve(audio[0], impulse_response[0])

    difference = output_np - output_tf.numpy()
    total_difference = np.abs(difference).mean()
    threshold = 1e-3
    assert total_difference <= threshold


@pytest.mark.parametrize(
    'gain',
    [1.0, 0.1]
)
def test_delay_compensation_corrects_group_delay(gain: float, audio: np.ndarray) -> None:
    # Create random signal to filter.
    output_np = gain * audio[0]
    n_frequencies = 1025
    window_size = 257

    magnitudes = gain * torch.ones([1, n_frequencies])
    impulse_response = core.frequency_impulse_response(magnitudes, window_size)
    output_tf = core.fft_convolve(torch.from_numpy(audio), impulse_response,
                                  padding='same')[0]

    difference = output_np - output_tf.numpy()
    total_difference = np.abs(difference).mean()
    threshold = 1e-3
    assert total_difference <= threshold


def test_fft_convolve_checks_batch_size(audio: np.ndarray) -> None:
    # Create random signals to convolve with different batch sizes.
    impulse_response = torch.cat(
        [torch.from_numpy(audio), torch.from_numpy(audio)], dim=0)

    with pytest.raises(ValueError):
        _ = core.fft_convolve(torch.from_numpy(audio), impulse_response)


@pytest.mark.parametrize(
    'padding',
    ['same', 'valid']
)
def test_fft_convolve_allows_valid_padding_arguments(padding: str,
                                                     audio: np.ndarray) -> None:
    result = core.fft_convolve(torch.from_numpy(audio),
                               torch.from_numpy(audio), padding=padding)
    assert result.shape[0] == 1


@pytest.mark.parametrize(
    'padding',
    ['', 'invalid']
)
def test_fft_convolve_disallows_invalid_padding_arguments(padding: str,
                                                          audio: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _ = core.fft_convolve(torch.from_numpy(audio),
                              torch.from_numpy(audio), padding=padding)


@pytest.mark.parametrize(
    'n_frames',
    [1010, 999]
)
def test_fft_convolve_checks_number_of_frames(n_frames: int, audio: np.ndarray) -> None:
    # Create random signals to convolve with same batch sizes.
    impulse_response = torch.randn([1, n_frames, audio.shape[1]])
    with pytest.raises(ValueError):
        _ = core.fft_convolve(torch.from_numpy(audio), impulse_response)


@pytest.mark.parametrize(
    'fft_size, window_size',
    [
        (2048, 0),
        (2048, 257),
        (1024, 22),
        (1024, 2048),
    ]
)
def test_frequency_impulse_response_gives_correct_size(fft_size: int,
                                                       window_size: int) -> None:
    # Create random signals to convolve.
    n_frequencies = fft_size // 2 + 1
    magnitudes = torch.rand((1, n_frequencies))

    impulse_response = core.frequency_impulse_response(magnitudes, window_size)

    target_size = fft_size
    if target_size > window_size >= 1:
        target_size = window_size
        is_even = target_size % 2 == 0
        target_size -= int(is_even)

    impulse_response_size = int(impulse_response.shape[-1])
    assert impulse_response_size == target_size


@pytest.mark.parametrize(
    'n_frequencies, n_frames, window_size',
    [
        (1025, 0, 0),
        (1025, 0, 257),
        (513, 1, 257),
        (513, 13, 257),
        (513, 1000, 257),
    ]
)
def test_frequency_filter_gives_correct_size(n_frequencies: int, n_frames: int,
                                             window_size: int, audio: np.ndarray) -> None:
    # Create transfer function.
    if n_frames > 0:
        magnitudes = np.random.uniform(size=(1, n_frames,
                                             n_frequencies)).astype(np.float32)
    else:
        magnitudes = np.random.uniform(size=(1, n_frequencies)).astype(np.float32)

    audio_out = core.frequency_filter(
        torch.from_numpy(audio),
        torch.from_numpy(magnitudes), window_size=window_size, padding='same')

    audio_out_size = int(audio_out.shape[-1])
    assert audio_out_size == audio.shape[1]
