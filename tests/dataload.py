import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

class RiceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_data(csv_path, test_size=0.2, val_size=0.5):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    y = df['Class'].values
    X = df.drop(columns='Class').values

    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=val_size)

    train_dataset = RiceDataset(x_train, y_train)
    val_dataset = RiceDataset(x_val, y_val)
    test_dataset = RiceDataset(x_test, y_test)

    return train_dataset, val_dataset, test_dataset
