import pytest
import torch
from src.model import MonModel

@pytest.fixture
def dummy_data():
    x = torch.randn(10, 10)
    y = torch.randint(0, 2, (10,1)).float()
    return x, y

@pytest.fixture
def model():
    return MonModel(input_dim=10, hidden_dim=5)
