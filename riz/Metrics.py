import torch
import matplotlib.pyplot as plt

def calculate_accuracy(y_pred, y_true, seuil=0.5):
    correct = ((y_pred >= seuil) == y_true).float()
    return correct.mean().item() * 100  # accuracy en %

def plot_metrics(train_loss, val_loss, train_acc, val_acc):

    fig, axs = plt.subplots(1, 2, figsize=(15,5))    
    axs[0].plot(train_loss, label="Train Loss")
    axs[0].plot(val_loss, label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    
    axs[1].plot(train_acc, label="Train Accuracy")
    axs[1].plot(val_acc, label="Val Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()
    plt.show()

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)

def load_model(model_class, path="model.pth", device="cpu", **model_kwargs):
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
