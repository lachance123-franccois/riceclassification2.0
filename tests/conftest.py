import pytest
import sys
import os
import torch


chemin= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chemin)
from riz.model import monModel

@pytest.fixture
def dummy_data():
    x = torch.randn(10, 10)
    y = torch.randint(0, 2, (10,1)).float()
    return x, y

@pytest.fixture
def model():
    return monModel(input_dim=10, hidden_dim=5)
