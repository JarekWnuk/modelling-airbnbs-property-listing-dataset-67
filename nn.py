import numpy as np
import pandas as pd
import tabular_data
import torch

class AirbnbNightlyPriceRegressionDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        with open("listing.csv", mode="r", encoding="utf8") as f:
            df = pd.read_csv(f)
            df.drop(["Unnamed: 19"], axis=1, inplace=True)
        self.clean_df = tabular_data.clean_tabular_data(df)
    
    def __getitem__(self, index):
        features, label = tabular_data.load_airbnb(self.clean_df, "Price_Night")
        print(features.columns)
        example_features = features.iloc[index] 
        example_label = label.iloc[index]
        example_features_tensor = torch.tensor(example_features)
        return (example_features_tensor, example_label)
    
    def __len__(self):
        return len(self.clean_df)
    
dataset = AirbnbNightlyPriceRegressionDataset()
print(dataset[2])
print(len(dataset))