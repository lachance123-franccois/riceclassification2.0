import torch
import pytest
from riz.model import MonModel

def test_forward_pass(dummy_data):
    x, y = dummy_data
    model = MonModel(input_dim=x.shape[1], hidden_dim=5)
    output = model(x)
    assert output.shape == y.shape, "La sortie doit avoir la même forme que les labels"
    assert (output >= 0).all() and (output <= 1).all(), "La sortie doit être entre 0 et 1"

def test_backward_pass(dummy_data):
    x, y = dummy_data
    model = MonModel(input_dim=x.shape[1], hidden_dim=5)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    output = model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
