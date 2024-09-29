import joblib
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import product
from modelling import save_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import model_selection
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from tabular_data import load_airbnb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


with open("listing_clean.csv", mode="r", encoding="utf8") as f:
    df = pd.read_csv(f)
    df.drop(["Unnamed: 0"], axis=1, inplace=True)

def display_confusion_matrix(confusion_matrix):
    cm_normalised = confusion_matrix / confusion_matrix.sum()
    display = ConfusionMatrixDisplay(cm_normalised)
    display.plot()
    plt.show()

def print_metrics(y_true, y_pred, name="input data"):
    f1 = f1_score(y_true, y_pred, average="micro")
    precission = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nMetrics for {name} \n\nF1 score: {f1}\nPrecission score: {precission}\nRecall score: {recall}\nAccuracy score: {accuracy}")

def custom_tune_classification_model_hyperparameters(
        model_class, X_train, y_train, X_validation, y_validation, X_test, y_test, hyperparams_dict: dict
        ):
    """
    Custom function for tuning model hyperparameters.
    Args:
        model_class (linear_model): classification model class
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
    test_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    validation_accuracy = accuracy_score(y_true=y_validation, y_pred=y_pred_val)
    model_metrics = {"test_accuracy" : test_accuracy, "validation_accuracy" : validation_accuracy}

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
        new_accuracy_test = accuracy_score(y_true=y_test, y_pred=y_pred_test) 
        new_accuracy_validation = accuracy_score(y_true=y_validation, y_pred=y_pred_val)

        if new_accuracy_validation < model_metrics["validation_accuracy"]: #if new model has better accuracy reuslts then it becomes the best model
            best_model = new_model
            best_hyperparams = hyperparams
            model_metrics["test_accuracy"] = new_accuracy_test
            model_metrics["validation_accuracy"] = new_accuracy_validation

    return best_model, best_hyperparams, model_metrics

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
        best_model, best_hyperparams, model_metrics = custom_tune_classification_model_hyperparameters(
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
    best_accuracy = 0
    for dir, subdir, file in os.walk(folder):
        if os.path.isfile(dir + "/metrics.json"):
            with open(dir + "/metrics.json", mode="r") as f:
                json_string = f.read()
                metrics_dict = json.loads(json_string)
                accuracy_from_metrics = metrics_dict["validation_accuracy"]
                if accuracy_from_metrics > best_accuracy:
                    best_accuracy = accuracy_from_metrics
                    best_metrics_dict = metrics_dict
                    best_performance_dir = dir
    best_model_path = best_performance_dir + "/model.joblib"
    best_model = joblib.load(best_model_path)
    with open(best_performance_dir + "/hyperparameters.json", mode="r") as f:
        json_string = f.read()
        best_hyperparams_dict = json.loads(json_string)
    print(f"The best performing model is the {type(best_model).__name__}. \nWith a validation accuracy score of: \
          {best_metrics_dict['validation_accuracy']}")
    return best_model, best_hyperparams_dict, best_metrics_dict

if __name__ == "__main__":
    features, label = load_airbnb(df, "Category")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, label, test_size=0.3)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

    models = [SGDClassifier, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier]
    SGD_hyperparams = {"loss": ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
                       "penalty": ['l2', 'l1'], "max_iter": [1000, 2000, 5000],
                        "learning_rate": ['constant', 'optimal', 'invscaling'], "eta0": [0.1, 0.2, 0.5, 1]}
    DTC_hyperparams = {"criterion": ["gini", "entropy", "log_loss"],
                       "min_samples_leaf" : [1, 2], "max_features" : [1, 2, 3],
                       "splitter" : ["best", "random"], "min_samples_split" : [2, 3, 4]}
    RFC_hyperparams = {"n_estimators" : [500, 1000, 2000],
                       "criterion": ["gini", "entropy", "log_loss"],
                       "min_samples_leaf" : [1, 2, 3], "max_features" : [1, 2, 3]}
    GBC_hyperparams = {"learning_rate" : [0.1, 0.2, 0.5], "n_estimators" : [500, 1000, 2000, 4000],
                       "criterion" : ['friedman_mse', 'squared_error']}
    hyperparam_dicts = [SGD_hyperparams, DTC_hyperparams, RFC_hyperparams, GBC_hyperparams]
    folder = "models/classification/logistic_regression"
    evaluate_all_models(models, hyperparam_dicts, folder)
    best_model, best_hyperparams, best_metrics = find_best_model(folder)

