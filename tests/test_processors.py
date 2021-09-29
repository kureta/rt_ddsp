import numpy as np
import torch

from rt_ddsp import processors


def test_output_is_correct() -> None:
    processor = processors.Add()
    x = torch.zeros((2, 3)) + 1.0
    y = torch.zeros((2, 3)) + 2.0

    output = processor(x, y)

    expected = np.zeros((2, 3), dtype=np.float32) + 3.0

    np.testing.assert_array_equal(expected, output.numpy())
