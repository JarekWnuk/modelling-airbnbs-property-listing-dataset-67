from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import tabular_data
import torch

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open("listing.csv", mode="r", encoding="utf8") as f:
            df = pd.read_csv(f)
            df.drop(["Unnamed: 19"], axis=1, inplace=True)
        self.clean_df = tabular_data.clean_tabular_data(df)
        self.features, self.label = tabular_data.load_airbnb(self.clean_df, "Price_Night")
    
    def __getitem__(self, index):
        example_label = torch.tensor(self.label.iloc[index])
        example_features = torch.tensor(self.features.iloc[index].values)
        return (example_features, example_label)
    
    def __len__(self):
        return len(self.clean_df)
    
class NN(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # define layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),  
            torch.nn.ReLU(), 
            torch.nn.Linear(4, 1),    
        )
        self.double()
    def forward(self, X):
        #return prediction
        return self.layers(X)

def train(model, data_loader):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.02)

    for batch in data_loader:
        train_loss = 0
        features, label = batch
        label = label.unsqueeze(1)
        label = label.double()
        prediction = model(features)
        train_loss = F.mse_loss(prediction, label)
        writer.add_scalar("Loss/train", train_loss, epoch)
        train_loss.backward()
        optimiser.step()
        optimiser.zero_grad()
    

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, label in data_loader:
            label = label.unsqueeze(1)
            prediction = model(features)
            loss = F.mse_loss(prediction, label)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    writer.add_scalar("Loss/evaluate", avg_loss, epoch)
    model.train()
    return avg_loss


if __name__ == "__main__":
    writer = SummaryWriter()
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset,batch_size=64, shuffle=False)
    model = NN()
    num_epochs = 15
    for epoch in range(num_epochs):
        train(model, train_loader)
        writer.flush()
        avg_loss = evaluate(model, val_loader)
        print(f"Average validation loss: {round(avg_loss, 0)}")