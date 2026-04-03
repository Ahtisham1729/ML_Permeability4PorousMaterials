"""Tests for train_inverse_model.py — bounding_loss (core loss function)."""

import torch
import pytest

from train_inverse_model import bounding_loss

N_FEATURES = 11
N_TARGETS = 3


# =====================================================================
# bounding_loss — core loss function, must be correct
# =====================================================================

class TestBoundingLoss:
    def test_zero_loss_inside_bounds(self):
        theta_min = torch.tensor([0.0, 0.0, 0.0])
        theta_max = torch.tensor([1.0, 1.0, 1.0])
        theta_pred = torch.tensor([[0.5, 0.5, 0.5]])
        assert bounding_loss(theta_pred, theta_min, theta_max).item() == 0.0

    def test_positive_loss_outside_bounds(self):
        theta_min = torch.tensor([0.0])
        theta_max = torch.tensor([1.0])
        assert bounding_loss(torch.tensor([[-0.5]]), theta_min, theta_max).item() > 0.0
        assert bounding_loss(torch.tensor([[1.5]]), theta_min, theta_max).item() > 0.0

    def test_quadratic_scaling(self):
        """Double the violation → 4x the loss (quadratic penalty)."""
        theta_min = torch.tensor([0.0])
        theta_max = torch.tensor([1.0])
        loss_1 = bounding_loss(torch.tensor([[-1.0]]), theta_min, theta_max)
        loss_2 = bounding_loss(torch.tensor([[-2.0]]), theta_min, theta_max)
        assert abs(loss_2.item() / loss_1.item() - 4.0) < 1e-5

    def test_gradient_flows(self):
        """Gradients must flow through for training to work."""
        theta_pred = torch.tensor([[-0.5]], requires_grad=True)
        loss = bounding_loss(theta_pred, torch.tensor([0.0]), torch.tensor([1.0]))
        loss.backward()
        assert theta_pred.grad is not None
