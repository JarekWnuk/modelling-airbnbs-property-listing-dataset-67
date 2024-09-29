import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from modelling import save_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import model_selection
from sklearn.linear_model import SGDClassifier
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
    f1_validation= f1_score(y_validation, y_pred_val, average="micro")
    model_metrics = {"test_accuracy" : test_accuracy, "validation_accuracy" : validation_accuracy, "validation_f1" : f1_validation}

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
        new_f1_validation = f1_score(y_validation, y_pred_val, average="micro")

        if new_accuracy_validation < model_metrics["validation_accuracy"]: #if new model has better accuracy reuslts then it becomes the best model
            best_model = new_model
            best_hyperparams = hyperparams
            model_metrics["test_accuracy"] = new_accuracy_test
            model_metrics["validation_accuracy"] = new_accuracy_validation
            model_metrics["validation_f1"] = new_f1_validation

    return best_model, best_hyperparams, model_metrics

if __name__ == "__main__":
    features, label = load_airbnb(df, "Category")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, label, test_size=0.3)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

    SGD_hyperparams = {"loss": ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
                       "penalty": ['l2', 'l1'], "max_iter": [1000, 2000, 5000],
                        "learning_rate": ['constant', 'optimal', 'invscaling'], "eta0": [0.1, 0.2, 0.5, 1] }
    best_model, best_hyperparams, metrics = custom_tune_classification_model_hyperparameters(
        SGDClassifier, X_train, y_train, X_validation, y_validation, X_test, y_test, SGD_hyperparams
        )
    folder = "models/classification/logistic_regression"
    save_model(folder,best_model, best_hyperparams, metrics)
    # new_model = SGDClassifier()
    # new_model.fit(X_train, y_train)
    # y_pred_train = new_model.predict(X_train)
    # y_pred_test = new_model.predict(X_test)

    # print_metrics(y_test, y_pred_test, "test set")
    # print_metrics(y_train, y_pred_train, "training set")

    # cm_test = confusion_matrix(y_test, y_pred_test)
    # display_confusion_matrix(cm_test)