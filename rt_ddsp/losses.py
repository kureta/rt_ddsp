import functools

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from rt_ddsp import spectral_ops


class SpectralLoss(nn.Module):
    def __init__(self,
                 fft_sizes=(2048, 1024, 512, 256, 128, 64),
                 loss_type='L1',
                 mag_weight=1.0,
                 delta_time_weight=0.0,
                 delta_freq_weight=0.0,
                 cumsum_freq_weight=0.0,
                 logmag_weight=0.0,
                 loudness_weight=0.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.delta_time_weight = delta_time_weight
        self.delta_freq_weight = delta_freq_weight
        self.cumsum_freq_weight = cumsum_freq_weight
        self.logmag_weight = logmag_weight
        self.loudness_weight = loudness_weight

        self.spectrogram_ops = []
        for size in self.fft_sizes:
            spectrogram_op = functools.partial(spectral_ops.compute_mag, size=size)
            self.spectrogram_ops.append(spectrogram_op)

    def forward(self, target_audio, audio, weights=None):
        loss = 0.0

        diff = spectral_ops.diff
        cumsum = torch.cumsum

        # Compute loss for each fft size.
        for loss_op in self.spectrogram_ops:
            target_mag = loss_op(target_audio)
            value_mag = loss_op(audio)

            # Add magnitude loss.
            if self.mag_weight > 0:
                loss += self.mag_weight * mean_difference(
                    target_mag, value_mag, self.loss_type, weights=weights)

            if self.delta_time_weight > 0:
                target = diff(target_mag, axis=1)
                value = diff(value_mag, axis=1)
                loss += self.delta_time_weight * mean_difference(
                    target, value, self.loss_type, weights=weights)

            if self.delta_freq_weight > 0:
                target = diff(target_mag, axis=2)
                value = diff(value_mag, axis=2)
                loss += self.delta_freq_weight * mean_difference(
                    target, value, self.loss_type, weights=weights)

            # TODO(kyriacos) normalize cumulative spectrogram
            if self.cumsum_freq_weight > 0:
                target = cumsum(target_mag, dim=2)
                value = cumsum(value_mag, dim=2)
                loss += self.cumsum_freq_weight * mean_difference(
                    target, value, self.loss_type, weights=weights)

            # Add logmagnitude loss, reusing spectrogram.
            if self.logmag_weight > 0:
                target = spectral_ops.safe_log(target_mag)
                value = spectral_ops.safe_log(value_mag)
                loss += self.logmag_weight * mean_difference(
                    target, value, self.loss_type, weights=weights)

        if self.loudness_weight > 0:
            target = spectral_ops.compute_loudness(target_audio, n_fft=2048,
                                                   use_torch=True)
            value = spectral_ops.compute_loudness(audio, n_fft=2048, use_torch=True)
            loss += self.loudness_weight * mean_difference(
                target, value, self.loss_type, weights=weights)

        return loss


def mean_difference(target, value, loss_type='L1', weights=None):
    difference = target - value
    weights = 1.0 if weights is None else weights
    loss_type = loss_type.upper()
    if loss_type == 'L1':
        return torch.mean(torch.abs(difference * weights))
    elif loss_type == 'L2':
        return torch.mean(difference ** 2 * weights)
    elif loss_type == 'COSINE':
        return (1. - F.cosine_similarity(target, value, dim=-1)) * weights
    else:
        raise ValueError(f'Loss type ({loss_type}), must be "L1", "L2", or "COSINE"')
