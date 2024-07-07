from itertools import product
import pandas as pd
import numpy as np
import tabular_data
from sklearn import model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, root_mean_squared_error


with open("listing.csv", mode="r", encoding="utf8") as f:
    df = pd.read_csv(f)
    df.drop(["Unnamed: 19"], axis=1, inplace=True)

cleaned_df = tabular_data.clean_tabular_data(df)
features, label = tabular_data.load_airbnb(cleaned_df, "Price_Night")
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, label, test_size=0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

def custom_tune_regression_model_hyperparameters(
        model_class, X_train, y_train, X_validation, y_validation, X_test, y_test, hyperparams_dict: dict
        ):
    #run initial model without any hyperparameter tuning
    initial_model = model_class()
    initial_hyperparams = hyperparams_dict
    initial_model.fit(X_train, y_train)
    y_pred_test = initial_model.predict(X_test)
    y_pred_val = initial_model.predict(X_validation)

    #calculate metrics for initial model
    RMSE_test = root_mean_squared_error(y_true=y_test, y_pred=y_pred_test)
    RMSE_validation = root_mean_squared_error(y_true=y_validation, y_pred=y_pred_val)
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

    #train models for each hyperparameter combination and calculate metrics
    for hyperparams in hyperparam_combination_dicts:
        new_model = model_class(**hyperparams)
        new_model.fit(X_train, y_train)
        y_pred_test = new_model.predict(X_test)
        y_pred_val = new_model.predict(X_validation)
        new_RMSE_test = root_mean_squared_error(y_true=y_test, y_pred=y_pred_test) 
        new_RMSE_validation = root_mean_squared_error(y_true=y_validation, y_pred=y_pred_val)
        new_R2_validation = r2_score(y_validation, y_pred_val)

        if new_RMSE_validation < model_metrics["validation_RMSE"]: #if new model has better RMSE reuslts then it becomes the best model
            best_model = new_model
            best_hyperparams = hyperparams
            model_metrics["test_RMSE"] = new_RMSE_test
            model_metrics["validation_RMSE"] = new_RMSE_validation
            model_metrics["validation_R2"] = new_R2_validation

    return best_model, best_hyperparams, model_metrics

def tune_regression_model_hyperparameters(model, X_train, y_train, hyperparams_dict: dict):
    grid_search = model_selection.GridSearchCV(model, hyperparams_dict)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

linear_model = SGDRegressor()
linear_model.fit(X_train, y_train)
y_pred_train = linear_model.predict(X_train)
y_pred_test = linear_model.predict(X_test)

R2_train = r2_score(y_train, y_pred_train)
R2_test = r2_score(y_test, y_pred_test)
RMSE_train = root_mean_squared_error(y_train, y_pred_train)
RMSE_test = root_mean_squared_error(y_test, y_pred_test)

hyperparams_dict ={"loss" : ["squared_error", "huber"],"max_iter" : [100, 500, 1000, 2000], "penalty" : ["l2", "l1"]}
best_model, best_hyperparams, model_metrics = custom_tune_regression_model_hyperparameters(
    SGDRegressor, X_train, y_train, X_validation, y_validation, X_test, y_test, hyperparams_dict
    )

# import matplotlib.pyplot as plt

# y_hat_validation = best_model.predict(X_validation)

# def plot_predictions(y_pred, y_true):
#     samples = len(y_pred)
#     plt.figure()
#     plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
#     plt.scatter(np.arange(samples), y_true, c='b', label='true labels', marker='x')
#     plt.legend()
#     plt.xlabel('Sample numbers')
#     plt.ylabel('Values')
#     plt.show()

# plot_predictions(y_hat_validation, y_validation)

best_params_grid_search = tune_regression_model_hyperparameters(SGDRegressor(), X_train, y_train, hyperparams_dict)
model_best_heparparams = SGDRegressor(**best_params_grid_search)
model_best_heparparams.fit(X_train, y_train)
best_model_prediction_test = model_best_heparparams.predict(X_test)
best_model_prediction_val = model_best_heparparams.predict(X_validation)

best_model_metrics = {}
grid_RMSE_test = root_mean_squared_error(y_test, best_model_prediction_test)
grid_RMSE_val= root_mean_squared_error(y_validation, best_model_prediction_val)
grid_R2_val = r2_score(y_validation, best_model_prediction_val)
best_model_metrics["test_RMSE"] = grid_RMSE_test
best_model_metrics["validation_RMSE"] = grid_RMSE_val
best_model_metrics["validation_R2"] = grid_R2_val

print("\nBest hyperparameters found in custom grid search:")
for key, value in best_hyperparams.items():
    print(f"{key} : {value}")

print("\nBest hyperparameters found from sklearn grid search:")
for key, value in best_params_grid_search.items():
    print(f"{key} : {value}")

print("\nMetrics for best model from custom grid search:")
for key, value in model_metrics.items():
    print(f"{key} : {value}")

print("\nMetrics for best model from sklearn grid search:")
for key, value in best_model_metrics.items():
    print(f"{key} : {value}")