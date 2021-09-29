import pytest
import torch

from rt_ddsp import effects


@pytest.fixture
def audio() -> torch.Tensor:
    return torch.zeros(3, 16000)


@pytest.fixture
def reverb() -> effects.Reverb:
    return effects.Reverb(reverb_length=100)


def test_output_shape_and_variables_are_correct(reverb: effects.Reverb,
                                                audio: torch.Tensor) -> None:
    output = reverb(audio)

    assert list(audio.shape) == list(output.shape)
    does_require_grad = [p.requires_grad for p in reverb.parameters()]
    assert len(does_require_grad) != 0
    assert all(does_require_grad)


def test_get_controls_returns_correct_keys(reverb: effects.Reverb,
                                           audio: torch.CharTensor) -> None:
    controls = reverb.get_controls(audio)

    assert list(controls.keys()) == ['audio', 'ir']
