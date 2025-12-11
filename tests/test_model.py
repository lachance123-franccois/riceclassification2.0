import torch
import pytest
import sys
import os



chemin= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chemin)
from riz.model import monModel

def test_forward_pass(dummy_data):
    x, y = dummy_data
    model = monModel(input_dim=x.shape[1], hidden_dim=5)
    output = model(x)
    assert output.shape == y.shape, "La sortie doit avoir la même forme que les labels"
    assert (output >= 0).all() and (output <= 1).all(), "La sortie doit être entre 0 et 1"

def test_backward_pass(dummy_data):
    x, y = dummy_data
    model = monModel(input_dim=x.shape[1], hidden_dim=5)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    output = model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
