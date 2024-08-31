import json
import joblib
from itertools import product
import pandas as pd
import os
import tabular_data
from sklearn import model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
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
    """
    Custom function for tuning model hyperparameters.
    Args:
        model_class (linear_model): regression model class
        X_train (pd.Dataframe): features for training
        y_train (pd.Dataframe): labels for training
        X_validation (pd.Dataframe): features for validation testing
        y_validation (pd.Dataframe): labels for validation testing
        X_test (pd.Dataframe): features for testing
        y_test (pd.Dataframe): labels for testing
        hyperparams_dict (dict): dictionary with hyperparameters to evaluate
    Returns:
        best_model (linear_model): model with the best metrics
        best_hyperparams (dict): dictionary containing best hyperparameters
        model_metrics (dict): metrics for model with best hyperparameters
    """
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

    #check if passed hyperparameters align with the model
    for hyperparam_type in hyperparams_dict.keys():
        if hasattr(initial_model, hyperparam_type):
            continue
        else:
            raise ValueError(f"Incorrect hyperparameter '{hyperparam_type}' for {model_class}")
    
    #generate combinations of all hyperparameters
    keys, values = zip(*initial_hyperparams.items())
    hyperparam_combination_dicts = [dict(zip(keys, v)) for v in product(*values)]
    
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
    """
    Function that tunes model hyperparameters using sklearn's GridSearchCV.
    Args:
        model (linear model): regression model class
        X_train (pd.Dataframe): features for training
        y_train (pd.Dataframe): labels for training
        hyperparams_dict (dict): dictionary with hyperparameters to evaluate
    Returns:
        grid_search.best_params_ (dict): dictionary containing best hyperparameters
    """
    grid_search = model_selection.GridSearchCV(model, hyperparams_dict)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def save_model(folder: str, model, hyperparams: dict, metrics: dict) -> None:
    """
    Function that alows to save the model to a joblib file, and it's hyperparameters and metrics to json files.
    Args:
        folder (str): path to save the file
        model (linear model): regression model instance
        hyperparams (dict): dictionary with hyperparameters
        metrics (dict): dictionary with model metrics

    """
    model_complete_path = folder + "/model.joblib"
    model_normalized_path = os.path.normcase(model_complete_path)
    joblib.dump(model, model_normalized_path)

    hyperparams_json_string = json.dumps(hyperparams)
    hyperparams_complete_path = folder + "/hyperparameters.json"
    hyperparams_normalized_path = os.path.normcase(hyperparams_complete_path)
    joblib.dump(hyperparams_json_string, hyperparams_normalized_path)

    metrics_json_string = json.dumps(metrics)
    metrics_complete_path = folder + "/metrics.json"
    metrics_normalized_path = os.path.normcase(metrics_complete_path)
    joblib.dump(metrics_json_string, metrics_normalized_path)

if __name__ == "__main__":
    linear_model = SGDRegressor()
    linear_model.fit(X_train, y_train)
    y_pred_train = linear_model.predict(X_train)
    y_pred_test = linear_model.predict(X_test)

    R2_train = r2_score(y_train, y_pred_train)
    R2_test = r2_score(y_test, y_pred_test)
    RMSE_train = root_mean_squared_error(y_train, y_pred_train)
    RMSE_test = root_mean_squared_error(y_test, y_pred_test)

    hyperparams_dict ={"loss" : ["squared_error", "huber"],"max_iter" : [500, 1000, 2000, 3000], "penalty" : ["l2", "l1"]}
    best_model, best_hyperparams, model_metrics = custom_tune_regression_model_hyperparameters(
        SGDRegressor, X_train, y_train, X_validation, y_validation, X_test, y_test, hyperparams_dict
        )

    best_params_grid_search = tune_regression_model_hyperparameters(SGDRegressor(), X_train, y_train, hyperparams_dict)
    model_best_hyperparams = SGDRegressor(**best_params_grid_search)
    model_best_hyperparams.fit(X_train, y_train)
    best_model_prediction_test = model_best_hyperparams.predict(X_test)
    best_model_prediction_val = model_best_hyperparams.predict(X_validation)

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

    folder = "models/regression/linear_regression"
    #save_model(folder, best_model, best_hyperparams, model_metrics)

    models = [SGDRegressor, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor]

    #TODO hyperparameter tuning for each model
    for model in models:
        new_model = model()
        new_model.fit(X_train, y_train)
        y_pred_test = new_model.predict(X_test)
        y_pred_val = new_model.predict(X_validation)
        R2_test = r2_score(y_test, y_pred_test)
        R2_val = r2_score(y_validation, y_pred_val)
        RMSE_test = root_mean_squared_error(y_test, y_pred_test)
        RMSE_val = root_mean_squared_error(y_validation, y_pred_val)
        print(f"\n{new_model} metrics: \nR2 test: {R2_test}\nR2 validation: {R2_val}\nRMSE test: {RMSE_test}\nRMSE validation: {RMSE_val}\n")

