import torch 
import os
import sys
chemin= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chemin)
from riz.model import monModel
import pytest


def test_training_loop(dummy_data, tmp_path):
    x, y = dummy_data
    model = monModel(input_dim=x.shape[1], nbr_hidden=5)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Une seule itération pour tester
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss doit être positive"
    assert y_pred.shape == y.shape, "erreur de dimension"
    
    model_path = tmp_path / "model_test.pth"
    torch.save(model.state_dict(), model_path)
    loaded_model = monModel(input_dim=x.shape[1], nbr_hidden=5)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    with torch.no_grad():
        y_pred_loaded = loaded_model(x)
    assert y_pred_loaded.shape == y.shape
