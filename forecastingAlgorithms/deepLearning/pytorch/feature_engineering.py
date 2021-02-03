from sklearn.feature_selection import mutual_info_regression
import pandas as pd



def pearson(df: pd.DataFrame):
    x = df.iloc[:-1]
    y = df['Close'].shift(-1).dropna()
    cor_target = abs(x.corrwith(y, method="pearson"))


    cor_target = cor_target.sort_values()

    cols = x.shape[1]
    threshold_features = cor_target.iloc[cols - 32:]

    selected_features = threshold_features.index.values

    x_d = x[selected_features]

    return x_d

def spearman(df: pd.DataFrame):
    x = df.iloc[:-1]
    y = df['Close'].shift(-1).dropna()

    cor_target = abs(x.corrwith(y, method="spearman"))

    cor_target = cor_target.sort_values()

    cols = x.shape[1]
    threshold_features = cor_target.iloc[cols-32:]

    selected_features = threshold_features.index.values

    x_d = x[selected_features]


    return x_d

def mutual_info(df: pd.DataFrame):
    x = df.iloc[:-1]
    y = df['Close'].shift(-1).dropna()

    mi_target = mutual_info_regression(x, y)

    mi = pd.Series(mi_target, index=x.columns.values)
    mi = mi.sort_values()

    cols = x.shape[1]
    mi = mi.iloc[cols-32:]

    selected_features = mi.index.values


    x_d = x[selected_features]

    return x_d