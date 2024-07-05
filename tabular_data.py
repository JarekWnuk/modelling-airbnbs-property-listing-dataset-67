import pandas as pd
import numpy as np


def remove_rows_with_missing_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    This method removes all rows that have missing entries in the ratings columns.
    Args:
        df (DataFrame): the unprocessed DataFrame
    Returns:
        df (DataFrame): the processed DataFrame
    """
    df.dropna(subset=["Cleanliness_rating", "Accuracy_rating", "Communication_rating",
                        "Location_rating", "Check-in_rating", "Value_rating"], inplace=True)
    return df

def combine_description_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    This method cleans descriptions of all the properties. This includes both removal of unwanted characters and replacing
    characters with spaces.
    Args:
        df (DataFrame): the unprocessed DataFrame
    Returns:
        df (DataFrame): the processed DataFrame
    """
    df.dropna(subset=["Description"], inplace=True)

    remove_chars = ["[", "'About this space", "', '", " ,'", "'", "]",  "'',", ", ,", "@ ", "''", " , "]
    for char in remove_chars:
        df["Description"] = df["Description"].str.replace(char, repl="")

    replace_with_space = ["\\n", "nn", "', '"]
    for char in replace_with_space:
        df["Description"] = df["Description"].str.replace(char, repl=" ")
    return df

def set_default_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some entries in the property database have missing entries in necessary fields. This method imputes the number 1
    in all such cases, considering the minimum available, for instance: number of beds.
    Args:
        df (DataFrame): the unprocessed DataFrame
    Returns:
        df (DataFrame): the processed DataFrame
    """
    cols_to_set = ["guests", "beds", "bathrooms", "bedrooms"]
    for column in cols_to_set:
        df[column] = df[column].fillna(1)
    df[pd.to_numeric(df["bedrooms"], errors='coerce').isnull()] = 1
    return df

def clean_tabular_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This methods uses other methods in sequence to clean the data. The "bedrooms" and "guests" columns are converted to int64.
    Args:
        df (DataFrame): the unprocessed DataFrame
    Returns:
        df (DataFrame): the processed DataFrame
    """
    df_removed_null_ratings = remove_rows_with_missing_ratings(df)
    df_ratings_combined = combine_description_strings(df_removed_null_ratings)
    df_features_set = set_default_feature_values(df_ratings_combined)
    df_features_set["bedrooms"] = df_features_set["bedrooms"].astype("int64")
    df_features_set["guests"] = df_features_set["guests"].astype("int64")
    return df_features_set

def load_airbnb(df: pd.DataFrame, label) -> tuple:
    """
    This method splits the data into a tuple with features and labels for use in machine learning.
    The name of the label column needs to be passed to the method. An exception is raised if the column does not exist
    in the DataFrame. Please ensure that the data has been cleaned prior to splitting.
    Args:
        df (DataFrame): DataFrame to split
        label (str): name of the label column
    Returns:
        features, labels (tuple): a tuple containing the features and labels extracted from the DataFrame
    """
    df = df.select_dtypes(include=np.number)
    if label not in df.columns:
        raise Exception("Label not found in columns!")
    labels = df.pop(label)
    features = df
    return (features, labels)

if __name__ == "__main__":

    with open("listing.csv", mode="r", encoding="utf8") as f:
        df = pd.read_csv(f)
        df.drop(["Unnamed: 19"], axis=1, inplace=True)
    
    df_clean = clean_tabular_data(df)
    features, labels = load_airbnb(df_clean,"bathrooms")
    print(features)
    pd.DataFrame.to_csv(df_clean, "listing_clean.csv")