import pandas as pd
import numpy as np

# test if value is number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# drop high cardinality columns
def DropHighCardinalityColumns(X):
    (m,n) = X.shape
    for c in X.columns:
        cardinality = X.loc[:,c].nunique()/m
        if cardinality > 0.98:
            X.drop(c, axis=1, inplace=True)
    return X

# replace blanks
def ReplaceBlanks(X):
    for column in X:
        # count number of values that are not numbers
        num_count = 0
        non_num_count = 0
        non_nums = []

        for index, item in X[column].iteritems():
            if is_number(item) == True:
                num_count += 1
            else:
                non_num_count += 1
                non_nums.append(item)

        non_nums = set(non_nums)

        # if mostly numeric, replace ' ' and '' with NaN and convert to numeric
        if non_num_count/(num_count+non_num_count) <=0.1:
            X[column].replace(" ", np.nan, inplace=True)
            X[column].replace("", np.nan, inplace=True)
            X[column] = pd.to_numeric(X[column])
    return X

# scale numeric fields
def ScaleNumerics(X):
    # get numeric fields
    num_cols = X.select_dtypes(include=[np.number]).copy()

    # use min and max of a column to scale numeric column
    for column in num_cols:
        min_val = min(X[column])
        max_val = max(X[column])
        rng = (max_val - min_val)
        col_normalized = X[column].add(-rng/2).divide(rng/2)
        X.drop(column, axis=1, inplace=True)
        X = pd.concat([X, col_normalized], axis=1)
    return X

# create dummies
def CreateDummies(X):
    # get non-numeric fields
    non_num_cols = X.select_dtypes(exclude=[np.number]).copy()

    # replace 'No phone service' and 'No internet service' with 'No'
    non_num_cols.replace("No phone service", "No", inplace=True)
    non_num_cols.replace("No internet service", "No", inplace=True)

    # get dummies
    for column in non_num_cols:
        new_cols = pd.get_dummies(non_num_cols[column], prefix=column, drop_first=True)
        X.drop(column, axis=1, inplace=True)
        X = pd.concat([X, new_cols], axis=1)
    return X

# Telco Preprocessing
def TelcoPreprocessing(X,y,X_only):
    # X
    # drop high cardinality columns
    DropHighCardinalityColumns(X)
    # replace blanks
    ReplaceBlanks(X)
    # scale numeric colums
    X = ScaleNumerics(X)
    # get dummies
    X = CreateDummies(X)

    if X_only == True:
        return X
    else:
        # y
        # get dummies
        y = pd.get_dummies(y, prefix='Churn', drop_first=True)
        return (X,y)
