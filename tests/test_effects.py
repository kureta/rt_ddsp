from datetime import timedelta

import librosa
import numpy as np
import torch
from hypothesis import strategies as st, given, settings

from rt_ddsp.effects import Reverb


def get_audio(sample_rate: int, batch_size: int) -> torch.Tensor:
    signal, _ = librosa.load(
        '/home/kureta/Music/violin/Violin Samples/yee_bach_dance_D#52.wav',
        sample_rate,
        mono=True,
    )
    signal: torch.Tensor = torch.from_numpy(signal[None, None, :])

    while signal.ndim < 3:
        signal.unsqueeze_(0)

    signal = signal.repeat(batch_size, 1, 1)

    return signal


@given(
    sample_rate=st.sampled_from([16000, 44100]),
    duration=st.sampled_from([1.0, 1.5, 2.756, 3.0]),
    batch_size=st.sampled_from([1, 5, 10]),
    live=st.booleans()
)
@settings(deadline=timedelta(milliseconds=8000))
def test_reverb_shape(sample_rate: int, duration: float, batch_size: int, live: bool) -> None:
    reverb = Reverb(sample_rate, duration, batch_size, live)
    reverb.ir.data[...] = 0.0
    reverb.ir.data[..., 0] = 1.0
    reverb.ir.data[..., sample_rate // 2] = 1.0

    audio = get_audio(sample_rate, batch_size)
    with torch.no_grad():
        result = reverb(audio)

    delay = torch.zeros_like(audio)
    delay[...] = audio
    delay[..., sample_rate//2:] += audio[..., :-sample_rate//2]

    assert audio.shape == result.shape
    np.testing.assert_allclose(result, delay, 1e-1, 1e-6)
