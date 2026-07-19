"""tests/test_two_tower.py — Unit tests for Two-Tower model and InfoNCE loss."""
import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.two_tower import MLP, UserTower, ItemTower, TwoTowerModel
from src.models.infonce_loss import infonce_loss

class TestMLP:
    def test_shape(self):
        out = MLP([64, 32, 16])(torch.randn(8, 64))
        assert out.shape == (8, 16)

class TestUserTower:
    def test_normalised(self):
        t   = UserTower(n_users=50, embed_dim=64, output_dim=32)
        out = t(torch.randint(0, 50, (8,)), torch.randn(8, 3))
        assert out.shape == (8, 32)
        assert torch.allclose(out.norm(dim=1), torch.ones(8), atol=1e-5)

class TestItemTower:
    def test_normalised(self):
        t   = ItemTower(n_items=100, n_categories=10, text_dim=32, embed_dim=64, output_dim=32)
        out = t(torch.randint(0,100,(8,)), torch.randint(0,10,(8,)), torch.randn(8,32), torch.randn(8,2))
        assert out.shape == (8, 32)
        assert torch.allclose(out.norm(dim=1), torch.ones(8), atol=1e-5)

class TestTwoTowerModel:
    def setup_method(self):
        self.m = TwoTowerModel(50, 100, 10, text_dim=32, embed_dim=64, output_dim=32)

    def test_forward(self):
        B   = 8
        u,v = self.m(
            torch.randint(0,50,(B,)), torch.randn(B,3),
            torch.randint(0,100,(B,)), torch.randint(0,10,(B,)),
            torch.randn(B,32), torch.randn(B,2),
        )
        assert u.shape == v.shape == (B, 32)
        assert torch.allclose(u.norm(dim=1), torch.ones(B), atol=1e-5)

    def test_temperature_positive(self):
        assert self.m.temperature.item() > 0

class TestInfoNCE:
    def test_positive(self):
        B   = 8
        u   = torch.nn.functional.normalize(torch.randn(B, 16), dim=1)
        v   = torch.nn.functional.normalize(torch.randn(B, 16), dim=1)
        loss = infonce_loss(u, v, torch.tensor(0.07))
        assert loss.item() > 0

    def test_perfect_alignment(self):
        e    = torch.nn.functional.normalize(torch.eye(8), dim=1)
        loss = infonce_loss(e, e, torch.tensor(0.07))
        assert loss.item() < 0.1

    def test_backward(self):
        B    = 8
        u_raw = torch.randn(B, 16, requires_grad=True)
        v_raw = torch.randn(B, 16, requires_grad=True)
        u = torch.nn.functional.normalize(u_raw, dim=1)
        v = torch.nn.functional.normalize(v_raw, dim=1)
        infonce_loss(u, v, torch.tensor(0.07)).backward()
        assert u_raw.grad is not None and v_raw.grad is not None
