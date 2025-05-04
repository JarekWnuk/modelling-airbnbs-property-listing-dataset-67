from torcheval.metrics.functional import r2_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import *
import datetime
import json
import numpy as np
import os
import pandas as pd
import tabular_data
import torch
import yaml


class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open("listing.csv", mode="r", encoding="utf8") as f:
            df = pd.read_csv(f)
            df.drop(["Unnamed: 19"], axis=1, inplace=True)
        self.clean_df = tabular_data.clean_tabular_data(df)
        self.features, self.label = tabular_data.load_airbnb(self.clean_df, "Price_Night")
    
    def __getitem__(self, index):
        example_label = self.label.iloc[index]
        example_features = torch.tensor(self.features.iloc[index].values)
        return (example_features, example_label)
    
    def __len__(self):
        return len(self.clean_df)
    
class NN(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        self.config = config
        super().__init__(*args, **kwargs)
        depth = config["model_depth"]
        hidden_layer_width = config["hidden_layer_width"]
        # define layers
        layers_list = []
        layers_list.append(torch.nn.Linear(11, hidden_layer_width))
        layers_list.append(torch.nn.ReLU())
        for x in range(depth):
            layers_list.append(torch.nn.Linear(hidden_layer_width, hidden_layer_width))
            layers_list.append(torch.nn.ReLU())
        layers_list.append(torch.nn.Linear(hidden_layer_width, 1))
        self.layers = torch.nn.Sequential(*layers_list)
        self.double()

    def forward(self, X):
        #return prediction
        return self.layers(X)
    
    def get_config(self):
        return self.config

def train(model, data_loader, hyperparam_dict):
    opt_dict = {"SGD": SGD, "Adagrad": Adagrad, "Adam": Adam, "Adamax": Adamax,
                "Adadelta": Adadelta, "Adam": AdamW, "ASGD": ASGD, "LBFGS": LBFGS,
                "NAdam": NAdam, "RAdam": RAdam, "RMSprop": RMSprop, "Rprop": Rprop,
                "SparseAdam": SparseAdam}
    if hyperparam_dict["optimiser"] in opt_dict.keys():
        opt = opt_dict[hyperparam_dict["optimiser"]]
    else:
        raise ValueError("The optimiser could not be found!")
    learning_rate = hyperparam_dict["learning_rate"]
    optimiser = opt(model.parameters(), lr=learning_rate)
    for batch in data_loader:
        RMSE_loss = 0
        features, label = batch
        label = label.unsqueeze(1)
        label = label.double()
        prediction = model(features)
        MSE_loss = F.mse_loss(prediction, label)
        RMSE_loss = torch.sqrt(MSE_loss)
        writer.add_scalar("Loss/train", RMSE_loss, epoch)
        MSE_loss.backward()
        optimiser.step() 
        optimiser.zero_grad()

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    total_pred_time = 0
    with torch.no_grad():
        for batch in data_loader:
            features, label = batch
            label = label.unsqueeze(1)
            pred_time_start = datetime.datetime.now()
            prediction = model(features)
            pred_time_end = datetime.datetime.now()
            pred_duration = pred_time_end - pred_time_start
            if total_pred_time == 0:
                total_pred_time = pred_duration
            total_pred_time += pred_duration

            loss = F.mse_loss(prediction, label)
            RMSE_loss = torch.sqrt(loss)
            total_loss += RMSE_loss.item()
            r2 = r2_score(prediction, label)

    avg_pred_time = total_pred_time / len(data_loader)
    avg_RMSE_loss = total_loss / len(data_loader)
    writer.add_scalar("Loss/evaluate", avg_RMSE_loss, epoch)
    model.train()
    return avg_RMSE_loss, r2, avg_pred_time

def get_nn_config():
    with open("nn_config.yaml", "r") as cf:
        nn_config_loaded = yaml.safe_load(cf)
    return nn_config_loaded

def save_model(model, model_metrics):
    if hasattr(model, "state_dict"):
        current_datetime = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        save_path = "models/neural_networks/regression/" + current_datetime 
        os.makedirs(save_path)
        normalized_save_path = os.path.normcase(save_path)
        torch.save(model.state_dict(), normalized_save_path + "/model.pt")
        # f = open(normalized_save_path + "/hyperparameters.json", "wb")
        # f.write(model.get_config())
        # f2 = open(normalized_save_path + "/metrics.json", "wb")
        # f2.write(model_metrics)
        hyperparams = model.get_config()
        hyperparams_path = normalized_save_path + "/hyperparameters.json"
        with open(hyperparams_path, "w") as file:
            json.dump(hyperparams, file)

        metrics_path = normalized_save_path + "/metrics.json"
        with open(metrics_path, "w") as file:
            json.dump(model_metrics, file)


if __name__ == "__main__":
    writer = SummaryWriter()
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset,batch_size=32, shuffle=False)

    model_metrics = {}
    nn_config = get_nn_config()
    model = NN(nn_config)
    num_epochs = 50

    training_start_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train(model, train_loader, nn_config)
        writer.flush()
    training_end_time = datetime.datetime.now()
    training_duration = training_end_time - training_start_time

    train_RMSE_loss, train_r2_score, avg_train_pred_time = evaluate(model, train_loader)
    val_RMSE_loss, val_r2_score, avg_val_pred_time = evaluate(model, val_loader)
    test_RMSE_loss, test_r2_score, avg_test_pred_time = evaluate(model, val_loader)

    inference_latency = (avg_train_pred_time + avg_val_pred_time + avg_test_pred_time) / 3
    model_metrics["training_duration"] = str(training_duration)
    model_metrics["RMSE_loss"] = [train_RMSE_loss, val_RMSE_loss, test_RMSE_loss]
    model_metrics["R_squared"] = [float(train_r2_score), float(val_r2_score), float(test_r2_score)]
    model_metrics["inference_latency"] = str(inference_latency)

    save_model(model, model_metrics)

    print(f"Average validation loss: {round(val_RMSE_loss, 0)}, R2: {val_r2_score}")
    print(f"Average train loss: {round(val_RMSE_loss, 0)}, R2: {val_r2_score}")
