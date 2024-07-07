from itertools import product
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

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_validation, y_validation, 
                                                 X_test, y_test, hyperparams_dict: dict):
    initial_model = model_class()
    initial_hyperparams = hyperparams_dict
    initial_model.fit(X_train, y_train)
    y_pred_test = initial_model.predict(X_test)
    y_pred_val = initial_model.predict(X_validation)
    RMSE_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test, squared=False)
    RMSE_validation = mean_squared_error(y_true=y_validation, y_pred=y_pred_val, squared=False)
    R2_validation= r2_score(y_validation, y_pred_val)
    model_metrics = {"test_RMSE" : RMSE_test, "validation_RMSE" : RMSE_validation, "validation_R2" : R2_validation}

    for hyperparam_type in hyperparams_dict.keys():
        if hasattr(initial_model, hyperparam_type): #check if passed hyperparameters align with the model
            continue
        else:
            raise ValueError(f"Incorrect hyperparameter '{hyperparam_type}' for {model_class}")
    
    keys, values = zip(*initial_hyperparams.items())
    hyperparam_combination_dicts = [dict(zip(keys, v)) for v in product(*values)] #generate combinations of all hyperparameters
    best_model = initial_model
    best_hyperparams = initial_hyperparams
    print(hyperparam_combination_dicts)
    for hyperparams in hyperparam_combination_dicts:
        new_model = model_class(**hyperparams)
        new_model.fit(X_train, y_train)
        y_pred_test = new_model.predict(X_test)
        y_pred_val = new_model.predict(X_validation)
        new_RMSE_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test, squared=False)
        new_RMSE_validation = mean_squared_error(y_true=y_validation, y_pred=y_pred_val, squared=False)
        new_R2_validation = r2_score(y_validation, y_pred_val)

        if new_RMSE_validation < model_metrics["validation_RMSE"]:
            best_model = new_model
            best_hyperparams = hyperparams
            model_metrics["test_RMSE"] = new_RMSE_test
            model_metrics["validation_RMSE"] = new_RMSE_validation
            model_metrics["validation_R2"] = new_R2_validation

    return best_model, best_hyperparams, model_metrics

linear_model = SGDRegressor()
linear_model.fit(X_train, y_train)
y_pred_train = linear_model.predict(X_train)
y_pred_test = linear_model.predict(X_test)

R2_train = r2_score(y_train, y_pred_train)
R2_test = r2_score(y_test, y_pred_test)
RMSE_train = mean_squared_error(y_train, y_pred_train, squared=False)
RMSE_test = mean_squared_error(y_test, y_pred_test, squared=False)

hyperparams_dict ={"loss" : ["squared_error", "huber"], "penalty" : ["l2", "l1"], "max_iter" : [100, 500, 1000, 2000]}
best_model, best_hyperparams, model_metrics = custom_tune_regression_model_hyperparameters(SGDRegressor, X_train, y_train,
                                                                                           X_validation, y_validation, X_test, y_test, hyperparams_dict)
print("\nBest hyperparameters found:")
for key, value in best_hyperparams.items():
    print(f"{key} : {value}")
print("\nMetrics for best model:")
for key, value in model_metrics.items():
    print(f"{key} : {value}")