from itertools import combinations
import pandas as pd
import numpy as np
import tabular_data
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor


with open("listing.csv", mode="r", encoding="utf8") as f:
    df = pd.read_csv(f)
    df.drop(["Unnamed: 19"], axis=1, inplace=True)

cleaned_df = tabular_data.clean_tabular_data(df)
features, label = tabular_data.load_airbnb(cleaned_df, "Price_Night")
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, label, test_size=0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

def custom_tune_regression_model_hyperparameters(model_class, train_set: pd.DataFrame, validation_set: pd.DataFrame, 
                                                 test_set: pd.DataFrame, hyperparams_dict: dict):
    initial_model = model_class()
    initial_hyperparams = hyperparams_dict
    metrics_for_best_model = {}

    for hyperparam_type in hyperparams_dict.keys():
        if hasattr(initial_model, hyperparam_type): #check if passed hyperparameters align with the model
            continue
        else:
            raise ValueError(f"Incorrect hyperparameter '{hyperparam_type}' for {model_class}")
    print(list(**initial_hyperparams))
    # for hyperparam_type in initial_hyperparams.keys():
    #     for hyperparam in hyperparam_type:


hyperparams_dict ={"loss" : "squared_error", "penalty" : ["l2", "l1"], "max_iter" : [100, 200, 300]}
linear_model = SGDRegressor()

custom_tune_regression_model_hyperparameters(SGDRegressor, train_set=X_train, validation_set=X_validation,
                                             test_set=X_test, hyperparams_dict=hyperparams_dict)


linear_model.fit(X_train, y_train)
y_pred_train = linear_model.predict(X_train)
y_pred_test = linear_model.predict(X_test)

R2_train = r2_score(y_train, y_pred_train)
R2_test = r2_score(y_test, y_pred_test)
RMSE_train = mean_squared_error(y_train, y_pred_train, squared=False)
RMSE_test = mean_squared_error(y_test, y_pred_test, squared=False)

print(f"R2 training set: {R2_train} \nR2 test set: {R2_test} \
       \nRMSE training set: {RMSE_train} \nRMSE test set: {RMSE_test}")