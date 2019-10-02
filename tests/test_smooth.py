import pytest
import torch

from sever.model import LabelSmoother


@pytest.mark.parametrize('tensor, eps, expected', [
    (
        torch.tensor([1., 0., 1., 0.]),
        1e-8,
        torch.tensor([
            (1 - 1e-8),
            (1e-8),
            (1 - 1e-8),
            (1e-8),
        ])
    ),
])
def test_smooth(tensor, eps, expected):
    smoother = LabelSmoother(eps)
    result = smoother(tensor)
    assert torch.allclose(result, expected)
