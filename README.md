# Modelling Airbnb's property listing dataset

## Description
The project focuses on data from the Airbnb database detailing property listings. Data will need to be cleaned and perpared for use in machine learning. Once completed this data will be used to train basic models imported from the sklearn library as well as a more complex neural network. The aim is to present the entire machine learning process from start to finish. The other goal is to show how models can be improved to get better predictions by tuning hyperparameters and comparing different types of models.
The project shows how the knowledge gained throughout the course can be used to create powerful and applicable tools.

## Installation instructions
The file containing the required dependencies is included, please use the following code to install packages:
```
 pip install -r requirements.txt
```
## Usage instructions
Use the following link to download a .zip file containing the dataset:
>https://aicore-project-files.s3.eu-west-1.amazonaws.com/airbnb-property-listings.zip.

Unzip it and you will find two folders: images and tabular_data. Inside the tabular_data folder, you will find a file called listing.csv which needs to be copied along with the images folder to the program directory.
Run tabular_data.py to clean the data.
Run modelling.py to get the best linear model with its hyperparameters and metrics.
Run classification.py to get the best classification model with its hyperparameters and metrics.

## File structure
After all files are in place and the program has been executed the file structure will be as below:
```
└── tabular_data.py
└── modelling.py
└── classification.py
└── requirements.txt
└── README.md
└── listing.csv
└── listing_clean.csv
└── 📁images # contains various folders and files
└── 📁models
    └── 📁linear_regression
        └── 📁DecisionTreeRegressor
            └── hyperparameters.json
            └── metrics.json
            └── model.joblib
        └── 📁GradientBoostingRegressor
            └── hyperparameters.json
            └── metrics.json
            └── model.joblib
        └── 📁RandomForestRegressor
            └── hyperparameters.json
            └── metrics.json
            └── model.joblib
        └── 📁SGDRegressor
            └── hyperparameters.json
            └── metrics.json
            └── model.joblib
    └── 📁classification
        └── 📁DecisionTreeClassifier
            └── hyperparameters.json
            └── metrics.json
            └── model.joblib
        └── 📁GradientBoostingClassifier
            └── hyperparameters.json
            └── metrics.json
            └── model.joblib
        └── 📁RandomForestClassifier
            └── hyperparameters.json
            └── metrics.json
            └── model.joblib
        └── 📁SGDClassifier
            └── hyperparameters.json
            └── metrics.json
            └── model.joblib
```
