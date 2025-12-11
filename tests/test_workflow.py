import torch
import os
import sys
chemin= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chemin)
from riz.model import monModel
from riz.Metrics import calcul_accuracy
import pytest

def test_training_workflow(dummy_data):
    x, y = dummy_data
    model = monModel(input_dim=x.shape[1], hidden_dim=5)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(2):  
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        acc = calcul_accuracy(y_pred, y)