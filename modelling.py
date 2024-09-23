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
    Creates directory specified in folder string if does not exist.
    Args:
        folder (str): path to save the file
        model (linear model): regression model instance
        hyperparams (dict): dictionary with hyperparameters
        metrics (dict): dictionary with model metrics
    """
    if os.path.isdir(folder) == False:
        os.mkdir(folder)

    model_complete_path = folder + "/model.joblib"
    model_normalized_path = os.path.normcase(model_complete_path)
    joblib.dump(model, model_normalized_path)

    hyperparams_complete_path = folder + "/hyperparameters.json"
    hyperparams_normalized_path = os.path.normcase(hyperparams_complete_path)
    with open(hyperparams_normalized_path, "w") as file:
        json.dump(hyperparams, file)

    metrics_complete_path = folder + "/metrics.json"
    metrics_normalized_path = os.path.normcase(metrics_complete_path)
    with open(metrics_normalized_path, "w") as file:
        json.dump(metrics, file)

def evaluate_all_models(models: list, hyperparam_dicts: list, folder: str) -> None:
    """
    Evaluates models with the passed hyperparameters using the custom tuning function.
    The list of model types and hyperparameter dicts must match in sequence.
    Saves the best model, best hyperparameters and metrics for each model type.
    Args:
        models (list): a list containing model classes
        hyperparam_dicts (list): a list containing hyperparameter dictionaries
        folder (str): directory used for saving data
    """
    models_and_hyperparams = zip(models, hyperparam_dicts)
    for model, hyperparams_dict in models_and_hyperparams:
        best_model, best_hyperparams, model_metrics = custom_tune_regression_model_hyperparameters(
                                                        model, X_train, y_train, X_validation, y_validation, X_test, y_test, hyperparams_dict)
        print(f"\n{type(best_model).__name__} \nHyperparameters: {best_hyperparams} \nMetrics: {model_metrics}")

        #use model name as directory name
        data_dir = folder + "/" + type(best_model).__name__
        save_model(data_dir, best_model, best_hyperparams, model_metrics)
    
def find_best_model(folder):
    """
    Function that traverses the passed directory and selects model with the best metrics.
    The search is focused on the metrics.json file. 
    Please ensure that the evaluate_all_models() function is run prior to this one.
    Args:
        folder (str): directory containing models and their metrics
    Returns:
        best_model (model instance): best model
        best_hyperparams_dict (dict): dictionary with best model hyperparameters
        best_metrics_dict (dict): dictionary with best model metrics
    """
    best_r2 = 0
    for dir, subdir, file in os.walk(folder):
        if os.path.isfile(dir + "/metrics.json"):
            with open(dir + "/metrics.json", mode="r") as f:
                json_string = f.read()
                metrics_dict = json.loads(json_string)
                r2_from_metrics = metrics_dict["validation_R2"]
                if r2_from_metrics > best_r2:
                    best_r2 = r2_from_metrics
                    best_metrics_dict = metrics_dict
                    best_performance_dir = dir
    best_model_path = best_performance_dir + "/model.joblib"
    best_model = joblib.load(best_model_path)
    with open(best_performance_dir + "/hyperparameters.json", mode="r") as f:
        json_string = f.read()
        best_hyperparams_dict = json.loads(json_string)
    print(f"The best performing model is the {type(best_model).__name__}. \nWith a validation R2 score of: {best_metrics_dict['validation_R2']}")
    return best_model, best_hyperparams_dict, best_metrics_dict
                    
if __name__ == "__main__":
    models = [SGDRegressor, DecisionTreeRegressor,  RandomForestRegressor, GradientBoostingRegressor]
    sgd_hyperparams_dict = {"loss" : ["squared_error", "huber"],"max_iter" : [500, 1000, 2000, 3000], "penalty" : ["l2", "l1"]}
    decision_tree_hyperparams_dict = {"criterion" : ["squared_error", "friedman_mse", "absolute_error"],
                                      "min_samples_leaf" : [1, 2], "max_features" : [1, 2, 3],
                                      "splitter" : ["best", "random"], "min_samples_split" : [2, 3, 4]}
    random_forest_hyperparams_dict = {"n_estimators" : [500, 1000, 2000],
                                      "criterion" : ["squared_error", "absolute_error"],
                                      "min_samples_leaf" : [1, 2, 3], "max_features" : [1, 2, 3]}
    grad_boost_hyperparams_dict = {"loss" : ['squared_error', 'absolute_error'],
                                   "learning_rate" : [0.1, 0.2, 0.5], "n_estimators" : [500, 1000, 2000],
                                   "criterion" : ['friedman_mse', 'squared_error']}
    hyperparam_dicts = [sgd_hyperparams_dict, decision_tree_hyperparams_dict, random_forest_hyperparams_dict, grad_boost_hyperparams_dict]
    folder = "models/regression/linear_regression"
    evaluate_all_models(models, hyperparam_dicts, folder)
    best_model, best_hyperparams, best_metrics = find_best_model(folder)
    
