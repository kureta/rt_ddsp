import torch

from rt_ddsp import synths


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
