import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import monModel
from dataload import load_data
from  Metrics import calcul_accuracy, plot_metrics, save_model
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cpu")
train_dataset, val_dataset, test_dataset = load_data(config["dataset_path"])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

input_dim = train_dataset.x.shape[1]
model = monModel(input_dim, config["hidden_neurons"]).to(device)
criterion = torch.nn.BCELoss()
optimizer = Adam(model.parameters(), lr=config["learning_rate"])

train_loss_plot, val_loss_plot = [], []
train_acc_plot, val_acc_plot = [], []

for epoch in range(config["epochs"]):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += ((y_pred >= 0.5) == y_batch).float().mean().item()
    total_loss /= len(train_loader)
    total_acc /= len(train_loader)
    train_loss_plot.append(total_loss)
    train_acc_plot.append(total_acc*100)

  
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()
            val_acc += ((y_pred >= 0.5) == y_batch).float().mean().item()
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_loss_plot.append(val_loss)
    val_acc_plot.append(val_acc*100)

    print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f} Acc={total_acc*100:.2f}%, "
          f"Val Loss={val_loss:.4f} Acc={val_acc*100:.2f}%")


acc = calcul_accuracy(y_pred, y_batch)
plot_metrics(train_loss_plot, val_loss_plot, train_acc_plot, val_acc_plot)
save_model(model, "model.pth")

