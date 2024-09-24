import matplotlib.pyplot as plt
import pandas as pd
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
    print(f"Metrics for {name} \n\nF1 score: {f1}\n\nPrecission score: {precission}\n\nRecall score: {recall}\n\nAccuracy score: {accuracy}")

features, label = load_airbnb(df, "Category")
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, label, test_size=0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

new_model = SGDClassifier()
new_model.fit(X_train, y_train)
y_pred_train = new_model.predict(X_train)
y_pred_test = new_model.predict(X_test)

f1_test = f1_score(y_test, y_pred_test, average="micro")
precission_test = precision_score(y_test, y_pred_test, average="micro")
recall_test = recall_score(y_test, y_pred_test, average="micro")
accuracy_test = accuracy_score(y_test, y_pred_test)

print_metrics(y_test, y_pred_test, "y_test")

cm_test = confusion_matrix(y_test, y_pred_test)
display_confusion_matrix(cm_test)