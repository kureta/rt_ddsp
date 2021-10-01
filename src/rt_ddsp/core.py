from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from scipy import fftpack

from rt_ddsp.types import AnyTensor


def torch_float32(x: AnyTensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.type(torch.float32)
    else:
        return torch.from_numpy(x.astype('float32'))


def exp_sigmoid(x: torch.Tensor,
                exponent: float = 10.0,
                max_value: float = 2.0,
                threshold: float = 1e-7) -> torch.Tensor:
    x = torch_float32(x)
    return max_value * torch.sigmoid(x) ** np.log(exponent) + threshold


def overlap_and_add(ys: torch.Tensor, hop_length: int) -> torch.Tensor:
    return torch.nn.functional.fold(
        ys.transpose(1, 2), (1, (ys.shape[1] - 1) * hop_length + ys.shape[2]),
        (1, ys.shape[2]), stride=(1, hop_length)
    ).squeeze(1).squeeze(1)


def get_harmonic_frequencies(frequencies: torch.Tensor,
                             n_harmonics: int) -> torch.Tensor:
    frequencies = torch_float32(frequencies)

    f_ratios = torch.linspace(1.0, float(n_harmonics), int(n_harmonics))
    f_ratios = f_ratios[None, None, :]
    harmonic_frequencies = frequencies * f_ratios
    return harmonic_frequencies


def remove_above_nyquist(frequency_envelopes: torch.Tensor,
                         amplitude_envelopes: torch.Tensor,
                         sample_rate: int = 16000) -> torch.Tensor:
    frequency_envelopes = torch_float32(frequency_envelopes)
    amplitude_envelopes = torch_float32(amplitude_envelopes)

    amplitude_envelopes = torch.where(
        torch.greater_equal(frequency_envelopes, sample_rate / 2.0),
        torch.zeros_like(amplitude_envelopes), amplitude_envelopes)
    return amplitude_envelopes


# TODO(discrepancy): using `align_corners` instead of `add_endpoint`
def resample(inputs: torch.Tensor,
             n_timesteps: int,
             method: str = 'linear',
             align_corners: bool = False) -> torch.Tensor:
    if method != 'linear':
        raise ValueError('Only linear interpolation is working for now.')
    inputs = inputs.permute(0, 2, 1)
    outputs = F.interpolate(inputs, n_timesteps, mode=method,
                            align_corners=align_corners)
    outputs = outputs.permute(0, 2, 1)
    return outputs


# TODO(discrepancy): Removed angular cumsum.
def oscillator_bank(frequency_envelopes: torch.Tensor,
                    amplitude_envelopes: torch.Tensor,
                    sample_rate: int = 16000,
                    sum_sinusoids: bool = True) -> torch.Tensor:
    frequency_envelopes = torch_float32(frequency_envelopes)
    amplitude_envelopes = torch_float32(amplitude_envelopes)

    # Don't exceed Nyquist.
    amplitude_envelopes = remove_above_nyquist(frequency_envelopes,
                                               amplitude_envelopes,
                                               sample_rate)

    # Angular frequency, Hz -> radians per sample.
    omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec
    omegas = omegas / float(sample_rate)  # rad / sample

    # Accumulate phase and synthesize.
    phases = torch.cumsum(omegas, dim=1)
    # TODO: Not in original
    phases %= 2 * np.pi

    # Convert to waveforms.
    waves = torch.sin(phases)
    audio = amplitude_envelopes * waves  # [mb, n_samples, n_sinusoids]
    if sum_sinusoids:
        audio = torch.sum(audio, dim=-1)  # [mb, n_samples]
    return audio


# TODO(discrepancy): Missing `upsample_with_windows` function.
def harmonic_synthesis(frequencies: torch.Tensor,
                       amplitudes: torch.Tensor,
                       harmonic_shifts: Optional[torch.Tensor] = None,
                       harmonic_distribution: Optional[torch.Tensor] = None,
                       n_samples: int = 64000,
                       sample_rate: int = 16000,
                       amp_resample_method: str = 'linear') -> torch.Tensor:
    frequencies = torch_float32(frequencies)
    amplitudes = torch_float32(amplitudes)

    if harmonic_distribution is not None:
        harmonic_distribution = torch_float32(harmonic_distribution)
        n_harmonics = int(harmonic_distribution.shape[-1])
    elif harmonic_shifts is not None:
        harmonic_shifts = torch_float32(harmonic_shifts)
        n_harmonics = int(harmonic_shifts.shape[-1])
    else:
        n_harmonics = 1

    # Create harmonic frequencies [batch_size, n_frames, n_harmonics].
    harmonic_frequencies = get_harmonic_frequencies(frequencies, n_harmonics)
    if harmonic_shifts is not None:
        harmonic_frequencies *= (1.0 + harmonic_shifts)

    # Create harmonic amplitudes [batch_size, n_frames, n_harmonics].
    if harmonic_distribution is not None:
        harmonic_amplitudes = amplitudes * harmonic_distribution
    else:
        harmonic_amplitudes = amplitudes

    # Create sample-wise envelopes.
    frequency_envelopes = resample(harmonic_frequencies,
                                   n_samples)  # cycles/sec
    amplitude_envelopes = resample(harmonic_amplitudes, n_samples,
                                   method=amp_resample_method)

    # Synthesize from harmonics [batch_size, n_samples].
    audio = oscillator_bank(frequency_envelopes,
                            amplitude_envelopes,
                            sample_rate=sample_rate)
    return audio


def apply_window_to_impulse_response(impulse_response: torch.Tensor,
                                     window_size: int = 0,
                                     causal: bool = False) -> torch.Tensor:
    impulse_response = torch_float32(impulse_response)

    # If IR is in causal form, put it in zero-phase form.
    if causal:
        impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    ir_size = int(impulse_response.shape[-1])
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = torch.hann_window(window_size)

    # Zero pad the window and put in in zero-phase form.
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:],
                            torch.zeros([padding]),
                            window[:half_idx]], dim=0)
    else:
        window = torch.fft.fftshift(window, dim=-1)

    # Apply the window, to get new IR (both in zero-phase form).
    window = torch.broadcast_to(window, impulse_response.shape)
    impulse_response = window * impulse_response

    # Put IR in causal form and trim zero padding.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat([impulse_response[..., first_half_start:],
                                      impulse_response[..., :second_half_end]],
                                     dim=-1)
    else:
        impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

    return impulse_response


def padding_end(signal: torch.Tensor, frame_size: int, hop_size: int) -> torch.Tensor:
    size = signal.shape[-1]
    diff = (size - frame_size) % hop_size
    if diff > 0:
        padding = hop_size - diff
        signal = F.pad(signal, (0, padding))
    return signal


def frame(signal: torch.Tensor,
          frame_size: int,
          hop_size: int,
          pad_end: bool = False) -> torch.Tensor:
    if pad_end:
        signal = padding_end(signal, frame_size, hop_size)

    return signal.unfold(1, frame_size, hop_size)


def get_fft_size(frame_size: int, ir_size: int,
                 power_of_2: bool = True) -> int:
    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        # Next power of 2.
        fft_size = int(2 ** np.ceil(np.log2(convolved_frame_size)))
    else:
        fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
    return fft_size


def crop_and_compensate_delay(audio: torch.Tensor, audio_size: int,
                              ir_size: int,
                              padding: str,
                              delay_compensation: int) -> torch.Tensor:
    # Crop the output.
    if padding == 'valid':
        crop_size = ir_size + audio_size - 1
    elif padding == 'same':
        crop_size = audio_size
    else:
        raise ValueError(
            f'Padding must be \'valid\' or \'same\', instead of {padding}.')

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = ((ir_size - 1) // 2 -
             1 if delay_compensation < 0 else delay_compensation)
    end = crop - start
    return audio[:, start:-end]


def fft_convolve(audio: torch.Tensor,
                 impulse_response: torch.Tensor,
                 padding: str = 'same',
                 delay_compensation: int = -1) -> torch.Tensor:
    audio, impulse_response = torch_float32(audio), torch_float32(
        impulse_response)

    # Get shapes of audio.
    batch_size, audio_size = audio.shape

    # Add a frame dimension to impulse response if it doesn't have one.
    ir_shape = impulse_response.shape
    if len(ir_shape) == 2:
        impulse_response = impulse_response[:, None, :]

    # Broadcast impulse response.
    if ir_shape[0] == 1 and batch_size > 1:
        impulse_response = torch.tile(impulse_response, [batch_size, 1, 1])

    # Get shapes of impulse response.
    ir_shape = impulse_response.shape
    batch_size_ir, n_ir_frames, ir_size = ir_shape

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError(f'Batch size of audio ({batch_size}) and impulse '
                         f'response ({batch_size_ir}) must be the same.')

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size
    audio_frames = frame(audio, frame_size, hop_size, pad_end=True)

    # Check that number of frames match.
    n_audio_frames = int(audio_frames.shape[1])
    if n_audio_frames != n_ir_frames:
        raise ValueError(
            f'Number of Audio frames ({n_audio_frames}) and impulse response '
            f'frames ({n_ir_frames}) do not match. For small hop size = '
            f'ceil(audio_size / n_ir_frames), number of impulse response '
            f'frames must be a multiple of the audio size.')

    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
    audio_fft = torch.fft.rfft(audio_frames, fft_size)
    ir_fft = torch.fft.rfft(impulse_response, fft_size)

    # Multiply the FFTs (same as convolution in time).
    audio_ir_fft = torch.multiply(audio_fft, ir_fft)

    # Take the iFFT to re-synthesize audio.
    audio_frames_out = torch.fft.irfft(audio_ir_fft)
    audio_out = overlap_and_add(audio_frames_out, hop_size)

    # Crop and shift the output audio.
    return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding,
                                     delay_compensation)


def frequency_impulse_response(magnitudes: torch.Tensor,
                               window_size: int = 0) -> torch.Tensor:
    # Get the IR (zero-phase form).
    magnitudes = magnitudes.type(torch.complex64)
    impulse_response = torch.fft.irfft(magnitudes)

    # Window and put in causal form.
    impulse_response = apply_window_to_impulse_response(impulse_response,
                                                        window_size)

    return impulse_response


def frequency_filter(audio: torch.Tensor,
                     magnitudes: torch.Tensor,
                     window_size: int = 0,
                     padding: str = 'same') -> torch.Tensor:
    impulse_response = frequency_impulse_response(magnitudes,
                                                  window_size=window_size)
    return fft_convolve(audio, impulse_response, padding=padding)
