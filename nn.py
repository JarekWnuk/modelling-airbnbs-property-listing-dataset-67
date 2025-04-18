from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
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

def train(model, data_loader, epochs=5):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch in data_loader:
            features, label = batch
            label = label.unsqueeze(1)
            label = label.double()
            prediction = model(features)
            loss = F.mse_loss(prediction, label)
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
        

if __name__ == "__main__":
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    model = NN()
    first_data = train_loader[1]
    print(first_data)
    #train(model, train_loader)
