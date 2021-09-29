import torch


# TODO(discrepancy): original has `pad_end` instead of `center`
def stft(audio: torch.Tensor,
         frame_size: int = 2048,
         overlap: float = 0.75,
         center: bool = True) -> torch.Tensor:
    assert frame_size * overlap % 2.0 == 0.0

    s = torch.stft(
        audio,
        n_fft=frame_size,
        hop_length=int(frame_size * (1.0 - overlap)),
        win_length=frame_size,
        center=center,
    )
    return s
