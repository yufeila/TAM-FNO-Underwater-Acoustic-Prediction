import torch

from tam_fno.models import FNO2d_FiLM


def test_tam_fno_output_shape():
    model = FNO2d_FiLM(modes1=4, modes2=4, width=8, time_dim=24, padding=2)
    x = torch.randn(2, 8, 16, 1)
    t = torch.randn(2, 24)

    y = model(x, t)

    assert y.shape == (2, 8, 16, 1)
