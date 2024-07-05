import pandas as pd
import numpy as np
import tabular_data
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor


with open("listing.csv", mode="r", encoding="utf8") as f:
    df = pd.read_csv(f)
    df.drop(["Unnamed: 19"], axis=1, inplace=True)

cleaned_df = tabular_data.clean_tabular_data(df)
features, label = tabular_data.load_airbnb(cleaned_df, "Price_Night")
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, label, test_size=0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5)

model = SGDRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(mean_squared_error(y_true=y_test, y_pred=y_pred))