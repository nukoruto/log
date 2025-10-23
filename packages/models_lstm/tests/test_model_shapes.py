import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

torch = pytest.importorskip("torch")  # noqa: N816

from models_lstm.model import LSTMEventPredictor  # noqa: E402


def test_lstm_model_forward_shapes() -> None:
    torch.manual_seed(0)
    model = LSTMEventPredictor(input_dim=5, hidden_dim=7, num_op_categories=3)
    inputs = torch.randn(2, 4, 5)

    op_probs, z_hat = model(inputs)

    assert op_probs.shape == (2, 4, 3)
    assert z_hat.shape == (2, 4)
    assert torch.allclose(
        op_probs.sum(dim=-1), torch.ones(2, 4), atol=1e-6, rtol=0.0
    ), "Operation probabilities must sum to 1 across categories."


def test_lstm_model_backward_gradients() -> None:
    torch.manual_seed(1)
    model = LSTMEventPredictor(input_dim=5, hidden_dim=11, num_op_categories=4)
    inputs = torch.randn(3, 2, 5)

    op_probs, z_hat = model(inputs)
    loss = op_probs.sum() + z_hat.sum()
    loss.backward()

    has_grad = any(param.grad is not None for param in model.parameters())
    assert has_grad, "Expected gradients for model parameters after backward pass."


def test_lstm_model_init_hidden_matches_batch_and_layers() -> None:
    model = LSTMEventPredictor(
        input_dim=5, hidden_dim=13, num_op_categories=2, num_layers=2
    )
    hidden = model.init_hidden(batch_size=6)

    assert hidden[0].shape == (2, 6, 13)
    assert hidden[1].shape == (2, 6, 13)
    assert torch.all(hidden[0] == 0.0)
    assert torch.all(hidden[1] == 0.0)
