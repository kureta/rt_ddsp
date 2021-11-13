"""
Implementation of Multi-Scale Spectral Loss as described in DDSP,
which is originally suggested in NSF (Wang et al., 2019)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torchaudio.transforms import Spectrogram


class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss.
    """

    def __init__(self, n_fft: int, alpha: float = 1.0, overlap: float = 0.75,
                 eps: float = 1e-7):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length

        self.stft = Spectrogram(
            self.n_fft, hop_length=self.hop_length, power=1, center=True
        )

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        s_true = self.stft(x_true)
        s_pred = self.stft(x_pred)

        linear_term = F.l1_loss(s_pred, s_true)
        log_term = F.l1_loss((s_true + self.eps).log2(), (s_pred + self.eps).log2())

        loss = linear_term + self.alpha * log_term
        return loss


class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.

    Usage ::

    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)

    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)
    """

    def __init__(
        self, n_ffts: list, alpha: float = 1.0, overlap: float = 0.75, eps: float = 1e-7
    ):
        super().__init__()
        self.losses = nn.ModuleList(
            [SSSLoss(n_fft, alpha, overlap, eps) for n_fft in n_ffts]
        )

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        losses = [loss(x_pred, x_true) for loss in self.losses]
        return sum(losses, torch.tensor(0.)).sum()
