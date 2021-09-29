# type: ignore
import numpy as np
import pytest
import torch

from rt_ddsp import spectral_ops

pytestmark = pytest.mark.skipif(True, reason='Not properly implemented yet')


# TODO: This module totally sucks.

def gen_np_sinusoid(frequency, amp, sample_rate, audio_len_sec):
    x = np.linspace(0, audio_len_sec, int(audio_len_sec * sample_rate))
    audio_sin = amp * (np.sin(2 * np.pi * frequency * x))
    return audio_sin


def gen_np_batched_sinusoids(frequency, amp, sample_rate, audio_len_sec,
                             batch_size):
    batch_sinusoids = [
        gen_np_sinusoid(frequency, amp, sample_rate, audio_len_sec)
        for _ in range(batch_size)
    ]
    return np.array(batch_sinusoids)


def test_stft_tf_and_np_are_consistent():
    amp = 1e-2
    audio = amp * (np.random.rand(64000).astype(np.float32) * 2.0 - 1.0)
    frame_size = 2048
    hop_size = 128
    overlap = 1.0 - float(hop_size) / frame_size
    pad_end = True

    s_np = spectral_ops.stft_np(
        audio, frame_size=frame_size, overlap=overlap, pad_end=pad_end)

    s_tf = spectral_ops.stft(
        torch.from_numpy(audio).unsqueeze(0), frame_size=frame_size, overlap=overlap,
        pad_end=pad_end)

    # TODO(jesseengel): The phase comes out a little different, figure out why.
    np.testing.assert_array_almost_equal(np.abs(s_np), np.abs(s_tf))


def test_stft_shape_is_correct():
    n_batch = 2
    n_time = 125
    n_freq = 100
    mag = torch.ones([n_batch, n_time, n_freq])

    diff = spectral_ops.diff
    delta_t = diff(mag, axis=1)
    assert delta_t.shape[1] == mag.shape[1] - 1
    delta_delta_t = diff(delta_t, axis=1)
    assert delta_delta_t.shape[1] == mag.shape[1] - 2
    delta_f = diff(mag, axis=2)
    assert delta_f.shape[2] == mag.shape[2] - 1
    delta_delta_f = diff(delta_f, axis=2)
    assert delta_delta_f.shape[2] == mag.shape[2] - 2


def test_loudness_tf_and_np_are_consistent():
    amp = 1e-2
    audio = amp * (np.random.rand(64000).astype(np.float32) * 2.0 - 1.0)
    frame_size = 2048
    frame_rate = 250

    ld_tf = spectral_ops.compute_loudness(
        audio, n_fft=frame_size, frame_rate=frame_rate, use_torch=True)

    ld_np = spectral_ops.compute_loudness(
        audio, n_fft=frame_size, frame_rate=frame_rate, use_torch=False)

    np.testing.assert_array_almost_equal(np.abs(ld_np), np.abs(ld_tf))


@pytest.mark.parametrize(
    'use_torch, num_dims',
    [
        (False, 1),
        (False, 2),
        (True, 1),
        (True, 2)
    ]
)
def test_pad_or_trim_vector_to_expected_length(use_torch, num_dims):
    vector_len = 10
    padded_vector_expected_len = 15
    trimmed_vector_expected_len = 4

    # Generate target vectors for testing
    vector = np.ones(vector_len) + np.random.uniform()
    num_pad = padded_vector_expected_len - vector_len
    target_padded = np.concatenate([vector, np.zeros(num_pad)])
    target_trimmed = vector[:trimmed_vector_expected_len]

    # Make a batch of target vectors
    if num_dims > 1:
        batch_size = 16
        vector = np.tile(vector, (batch_size, 1))
        target_padded = np.tile(target_padded, (batch_size, 1))
        target_trimmed = np.tile(target_trimmed, (batch_size, 1))

    vector_padded = spectral_ops.pad_or_trim_to_expected_length(
        vector, padded_vector_expected_len, use_torch=use_torch)
    vector_trimmed = spectral_ops.pad_or_trim_to_expected_length(
        vector, trimmed_vector_expected_len, use_torch=use_torch)
    np.testing.assert_allclose(target_padded, vector_padded)
    np.testing.assert_allclose(target_trimmed, vector_trimmed)


@pytest.fixture
def compute_features_params():
    amp = 0.75
    frequency = 440.0
    frame_rate = 250

    return amp, frequency, frame_rate


@pytest.mark.parametrize(
    'sample_rate, audio_len_sec',
    [
        (16000, .21),
        (24000, .21),
        (48000, .21),
        (16000, .4),
        (24000, .4),
        (48000, .4),
    ]
)
def test_compute_loudness_at_sample_rate_1d(sample_rate, audio_len_sec,
                                            compute_features_params):
    amp, frequency, frame_rate = compute_features_params
    audio_sin = gen_np_sinusoid(frequency, amp, sample_rate,
                                audio_len_sec)
    expected_loudness_len = int(frame_rate * audio_len_sec)

    for use_torch in [False, True]:
        loudness = spectral_ops.compute_loudness(
            audio_sin, sample_rate, frame_rate, use_torch=use_torch)
        assert len(loudness) == expected_loudness_len
        assert np.all(np.isfinite(loudness))


@pytest.mark.parametrize(
    'sample_rate, audio_len_sec',
    [
        (16000, .21),
        (24000, .21),
        (48000, .21),
        (16000, .4),
        (24000, .4),
        (48000, .4),
    ]
)
def test_compute_loudness_at_sample_rate_2d(sample_rate, audio_len_sec,
                                            compute_features_params):
    amp, frequency, frame_rate = compute_features_params
    batch_size = 8
    audio_sin_batch = gen_np_batched_sinusoids(frequency, amp,
                                               sample_rate, audio_len_sec,
                                               batch_size)
    expected_loudness_len = int(frame_rate * audio_len_sec)

    for use_torch in [False, True]:
        loudness_batch = spectral_ops.compute_loudness(
            audio_sin_batch, sample_rate, frame_rate, use_torch=use_torch)

        assert loudness_batch.shape[0] == batch_size
        assert loudness_batch.shape[1] == expected_loudness_len
        assert np.all(np.isfinite(loudness_batch))

        # Check if batched loudness is equal to equivalent single computations
        audio_sin = gen_np_sinusoid(frequency, amp, sample_rate,
                                    audio_len_sec)
        loudness_target = spectral_ops.compute_loudness(
            audio_sin, sample_rate, frame_rate, use_torch=use_torch)
        loudness_batch_target = np.tile(loudness_target, (batch_size, 1))
        # Allow tolerance within 1dB
        np.testing.assert_allclose(loudness_batch, loudness_batch_target, atol=1, rtol=1)


@pytest.mark.parametrize(
    'sample_rate, audio_len_sec',
    [
        (16000, .21),
        (24000, .21),
        (48000, .21),
        (16000, .4),
        (24000, .4),
        (48000, .4),
    ]
)
def test_tf_compute_loudness_at_sample_rate(sample_rate, audio_len_sec,
                                            compute_features_params):
    amp, frequency, frame_rate = compute_features_params
    audio_sin = gen_np_sinusoid(frequency, amp, sample_rate,
                                audio_len_sec)
    loudness = spectral_ops.compute_loudness(audio_sin, sample_rate,
                                             frame_rate)
    expected_loudness_len = int(frame_rate * audio_len_sec)
    assert len(loudness) == expected_loudness_len
    assert np.all(np.isfinite(loudness))


@pytest.mark.parametrize(
    'sample_rate, audio_len_sec',
    [
        (44100, .21),
        (44100, .4),
    ]
)
def test_compute_loudness_indivisible_rates_raises_error(sample_rate, audio_len_sec,
                                                         compute_features_params):
    amp, frequency, frame_rate = compute_features_params
    audio_sin = gen_np_sinusoid(frequency, amp, sample_rate,
                                audio_len_sec)

    for use_torch in [False, True]:
        with pytest.raises(ValueError):
            spectral_ops.compute_loudness(
                audio_sin, sample_rate, frame_rate, use_torch=use_torch)
