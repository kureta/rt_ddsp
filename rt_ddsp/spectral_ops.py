from typing import Iterable

import librosa
import numpy as np
import torch
import torch.nn.functional as F  # noqa

from rt_ddsp.core import torch_float32
from rt_ddsp.types import AnyTensor

LD_RANGE = 120.0  # dB

# TODO: Get rid of dynamic module use for the sake of type safety.
#       In general, it might be better if we try to make everything work with both numpy
#       arrays and pytorch tensors.


def safe_log(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)


def amplitude_to_db(amplitude: AnyTensor,
                    use_torch: bool = False) -> torch.Tensor:
    """Converts amplitude to decibels."""
    lib = torch if use_torch else np
    amplitude = torch_float32(amplitude) if use_torch else amplitude
    amin = torch.tensor(1e-20) if use_torch else 1e-20  # Avoid log(0) instabilities.

    db = lib.log10(lib.maximum(amin, amplitude))  # type: ignore
    db *= 20.0
    return db


# TODO(discrepancy): Not sure if stft centered=True or not
def stft(audio: torch.Tensor,
         frame_size: int = 2048,
         overlap: float = 0.75,
         pad_end: bool = True) -> torch.Tensor:
    assert frame_size * overlap % 2.0 == 0.0
    hop_size = int(frame_size * (1.0 - overlap))
    if pad_end:
        n_samples_initial = int(audio.shape[-1])
        n_frames = int(np.ceil(n_samples_initial / hop_size))
        n_samples_final = (n_frames - 1) * hop_size + frame_size
        pad = n_samples_final - n_samples_initial
        padding = (0, pad)
        audio = F.pad(audio, padding, 'constant')

    s = torch.stft(
        audio,
        n_fft=frame_size,
        hop_length=hop_size,
        win_length=frame_size,
        center=False,
        return_complex=True
    ).abs().transpose(1, 2)
    return s


def stft_np(audio: np.ndarray, frame_size: int = 2048,
            overlap: float = 0.75, pad_end: bool = True) -> np.ndarray:
    assert frame_size * overlap % 2.0 == 0.0
    hop_size = int(frame_size * (1.0 - overlap))
    is_2d = (len(audio.shape) == 2)

    if pad_end:
        n_samples_initial = int(audio.shape[-1])
        n_frames = int(np.ceil(n_samples_initial / hop_size))
        n_samples_final = (n_frames - 1) * hop_size + frame_size
        pad = n_samples_final - n_samples_initial
        padding = ((0, 0), (0, pad)) if is_2d else ((0, pad),)
        audio = np.pad(audio, padding, 'constant')

    def stft_fn(y: np.ndarray) -> np.ndarray:
        return librosa.stft(y=y,
                            n_fft=int(frame_size),
                            hop_length=hop_size,
                            center=False).T

    s = np.stack([stft_fn(a) for a in audio]) if is_2d else stft_fn(audio)
    return s


# TODO(discrepancy): original has `pad_end` instead of `center`
def compute_mag(audio: torch.Tensor, size: int = 2048, overlap: float = 0.75,
                pad_end: bool = True) -> torch.Tensor:
    mag = torch.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))
    return torch_float32(mag)


def tensor_slice(x: torch.Tensor, begin: Iterable[int], size: Iterable[int]) -> torch.Tensor:
    end = [b + s for b, s in zip(begin, size)]
    return x[[slice(b, e) for b, e in zip(begin, end)]]


def diff(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shape = list(x.shape)
    ndim = len(shape)
    if axis >= ndim:
        raise ValueError(f'Invalid axis index: {axis} for tensor with only {ndim} axes.')

    begin_back = [0 for _ in range(ndim)]
    begin_front = [0 for _ in range(ndim)]
    begin_front[axis] = 1

    shape[axis] -= 1
    slice_front = tensor_slice(x, begin_front, shape)
    slice_back = tensor_slice(x, begin_back, shape)
    d = slice_front - slice_back
    return d


def pad_or_trim_to_expected_length(vector: AnyTensor,
                                   expected_len: int,
                                   pad_value: float = 0.0,
                                   len_tolerance: int = 20,
                                   use_torch: bool = False) -> AnyTensor:
    vector = torch_float32(vector) if use_torch else vector
    expected_len = int(expected_len)
    vector_len = int(vector.shape[-1])

    if abs(vector_len - expected_len) > len_tolerance:
        # Ensure vector was close to expected length to begin with
        raise ValueError('Vector length: {} differs from expected length: {} '
                         'beyond tolerance of : {}'.format(vector_len,
                                                           expected_len,
                                                           len_tolerance))
    # Pick tensorflow or numpy.
    lib = torch if use_torch else np

    is_1d = (len(vector.shape) == 1)
    vector = vector[None, :] if is_1d else vector

    # Pad missing samples
    if vector_len < expected_len:
        n_padding = expected_len - vector_len
        vector = lib.pad(  # type: ignore
            vector, ((0, 0), (0, n_padding)),
            mode='constant',
            constant_values=pad_value)
    # Trim samples
    elif vector_len > expected_len:
        vector = vector[..., :expected_len]

    # Remove temporary batch dimension.
    vector = vector[0] if is_1d else vector
    return vector


def compute_loudness(audio: AnyTensor,
                     sample_rate: int = 16000,
                     frame_rate: int = 250,
                     n_fft: int = 2048,
                     range_db: float = LD_RANGE,
                     ref_db: float = 20.7,
                     use_torch: bool = False) -> AnyTensor:
    if sample_rate % frame_rate != 0:
        raise ValueError(
            f'frame_rate: {frame_rate} must evenly divide sample_rate: {sample_rate}.'
            'For default frame_rate: 250Hz, suggested sample_rate: 16kHz or 48kHz')

    # Pick tensorflow or numpy.
    lib = torch if use_torch else np

    # Make inputs tensors for tensorflow.
    audio = torch_float32(audio) if use_torch else audio

    # Temporarily a batch dimension for single examples.
    is_1d = len(audio.shape) == 1
    audio = audio[None, :] if is_1d else audio

    # Take STFT.
    hop_size = sample_rate // frame_rate
    overlap = 1 - hop_size / n_fft
    stft_fn = stft if use_torch else stft_np
    s = stft_fn(audio, frame_size=n_fft, overlap=overlap, pad_end=True)  # type: ignore

    # Compute power.
    amplitude = lib.abs(s)  # type: ignore
    power_db = amplitude_to_db(amplitude, use_torch=use_torch)

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)[None, None, :]
    loudness = power_db + a_weighting

    # Set dynamic range.
    loudness -= ref_db
    loudness = lib.maximum(loudness, -range_db)  # type: ignore
    mean = torch.mean if use_torch else np.mean

    # Average over frequency bins.
    loudness = mean(loudness, axis=-1)  # type: ignore

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness

    # Compute expected length of loudness vector
    n_secs = audio.shape[-1] / float(
        sample_rate)  # `n_secs` can have milliseconds
    expected_len = int(n_secs * frame_rate)

    # Pad with `-range_db` noise floor or trim vector
    loudness = pad_or_trim_to_expected_length(
        loudness, expected_len, -range_db, use_torch=use_torch)
    return loudness
