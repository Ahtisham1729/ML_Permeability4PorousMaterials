"""Tests for train_inverse_model.py — InverseMLP, bounding_loss, to_physical_K."""

import numpy as np
import torch
import torch.nn as nn
import pytest
from sklearn.preprocessing import MinMaxScaler

from train_inverse_model import InverseMLP, bounding_loss, to_physical_K
N_FEATURES = 11
N_TARGETS = 3


# =====================================================================
# InverseMLP
# =====================================================================

class TestInverseMLP:
    def test_output_shape(self):
        model = InverseMLP(n_inputs=N_TARGETS, n_outputs=N_FEATURES,
                           hidden_layers=[16, 32, 16], dropout_rate=0.0)
        out = model(torch.randn(8, N_TARGETS))
        assert out.shape == (8, N_FEATURES)

    def test_default_hidden_layers(self):
        model = InverseMLP(n_inputs=N_TARGETS, n_outputs=N_FEATURES)
        # Default [256,512,512,512,256] → 5 hidden + 1 output = 6 Linear layers
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 6

    def test_custom_hidden_layers(self):
        model = InverseMLP(n_inputs=N_TARGETS, n_outputs=N_FEATURES,
                           hidden_layers=[16, 32, 16])
        out = model(torch.randn(4, N_TARGETS))
        assert out.shape == (4, N_FEATURES)

    def test_has_layernorm(self):
        model = InverseMLP(n_inputs=N_TARGETS, n_outputs=N_FEATURES,
                           hidden_layers=[16, 16])
        layernorms = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
        assert len(layernorms) > 0

    def test_no_nan_output(self):
        model = InverseMLP(n_inputs=N_TARGETS, n_outputs=N_FEATURES,
                           hidden_layers=[16, 16], dropout_rate=0.0)
        model.eval()
        out = model(torch.randn(8, N_TARGETS))
        assert torch.isfinite(out).all()


# =====================================================================
# bounding_loss
# =====================================================================

class TestBoundingLoss:
    def test_inside_bounds_zero_loss(self):
        theta_min = torch.tensor([0.0, 0.0, 0.0])
        theta_max = torch.tensor([1.0, 1.0, 1.0])
        theta_pred = torch.tensor([[0.5, 0.5, 0.5]])
        loss = bounding_loss(theta_pred, theta_min, theta_max)
        assert loss.item() == 0.0

    def test_below_min_positive_loss(self):
        theta_min = torch.tensor([0.0])
        theta_max = torch.tensor([1.0])
        theta_pred = torch.tensor([[-0.5]])
        loss = bounding_loss(theta_pred, theta_min, theta_max)
        assert loss.item() > 0.0

    def test_above_max_positive_loss(self):
        theta_min = torch.tensor([0.0])
        theta_max = torch.tensor([1.0])
        theta_pred = torch.tensor([[1.5]])
        loss = bounding_loss(theta_pred, theta_min, theta_max)
        assert loss.item() > 0.0

    def test_symmetric_penalty(self):
        theta_min = torch.tensor([-1.0])
        theta_max = torch.tensor([1.0])
        # 0.5 below min vs 0.5 above max
        pred_below = torch.tensor([[-1.5]])
        pred_above = torch.tensor([[1.5]])
        loss_below = bounding_loss(pred_below, theta_min, theta_max)
        loss_above = bounding_loss(pred_above, theta_min, theta_max)
        assert abs(loss_below.item() - loss_above.item()) < 1e-7

    def test_quadratic_scaling(self):
        theta_min = torch.tensor([0.0])
        theta_max = torch.tensor([1.0])
        # Violation of 1.0 vs 2.0 → loss ratio should be 1:4
        pred_1 = torch.tensor([[-1.0]])  # 1.0 below min
        pred_2 = torch.tensor([[-2.0]])  # 2.0 below min
        loss_1 = bounding_loss(pred_1, theta_min, theta_max)
        loss_2 = bounding_loss(pred_2, theta_min, theta_max)
        assert abs(loss_2.item() / loss_1.item() - 4.0) < 1e-5

    def test_gradient_flows(self):
        theta_pred = torch.tensor([[-0.5]], requires_grad=True)
        theta_min = torch.tensor([0.0])
        theta_max = torch.tensor([1.0])
        loss = bounding_loss(theta_pred, theta_min, theta_max)
        loss.backward()
        assert theta_pred.grad is not None


# =====================================================================
# to_physical_K
# =====================================================================

class TestToPhysicalK:
    def _make_scaler(self):
        """Fit a scaler on log10-transformed K values."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        k_log = np.array([[-10, -8, -6], [-2, -3, -4]], dtype=np.float64)
        scaler.fit(k_log)
        return scaler

    def test_with_log(self):
        scaler = self._make_scaler()
        k_scaled = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        result = to_physical_K(k_scaled, scaler, use_log=True)
        # Should be 10^(inverse_transform(scaled))
        k_model = scaler.inverse_transform(k_scaled)
        expected = np.power(10.0, k_model)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_without_log(self):
        scaler = self._make_scaler()
        k_scaled = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        result = to_physical_K(k_scaled, scaler, use_log=False)
        expected = scaler.inverse_transform(k_scaled)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_shape_preserved(self):
        scaler = self._make_scaler()
        k_scaled = np.random.default_rng(0).uniform(0, 1, (10, 3))
        result = to_physical_K(k_scaled, scaler, use_log=True)
        assert result.shape == (10, 3)
