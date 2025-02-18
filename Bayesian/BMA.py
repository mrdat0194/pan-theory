import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
from pycaret.datasets import get_data

def get_dataset(dataset_name):
    """Get dataset."""

    target_name = "SalePrice"
    df = get_data(dataset_name, verbose=False).reset_index(drop=True)
    X, y = df.drop(target_name, axis=1), df[target_name]
    # y = y.replace({yval: binval for yval, binval in zip(y.value_counts().index, [0,1])}).iloc[:,0]
    num_features = X.columns[X.apply(is_numeric)].tolist()
    cat_features = [c for c in X.columns if c not in num_features]
    X[cat_features] = X[cat_features].fillna({feature: "NULL" for feature in cat_features})
    return X, y, num_features, cat_features


def is_numeric(x):
    """Check whether an object is numeric."""
    try:
        x + 0
        return True
    except:
        return False


def float2kdollars(value, tick_number=None):
    return f"$ {int(round(value / 1000))}{'k' if abs(value) > 0 else ''}"

X_full, y_full, num_features, cat_features = get_dataset("house")
X_full[["square feet", "overall condition"]] = X_full[["GrLivArea","OverallCond"]]

X = X_full[["square feet"]]
y = y_full

regr = OLS(y, add_constant(X)).fit()
print(regr.summary())

# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
sns.pairplot(df)