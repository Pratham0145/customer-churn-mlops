# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):

    df = df.copy()

    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"].fillna(
        df["TotalCharges"].median(),
        inplace=True
    )

    le = LabelEncoder()

    df["Churn"] = le.fit_transform(df["Churn"])

    df = pd.get_dummies(
        df,
        drop_first=True
    )

    return df