from sklearn.feature_selection import SelectKBest, mutual_info_regression
from System import *



def pearson(x: pd.DataFrame, y:  pd.Series, system: SystemComponents):

    cor_target = abs(x.corrwith(y, method="pearson"))


    cor_target = cor_target.sort_values()

    features = cor_target[len(cor_target)-dimension:]

    selected_features = features.index.values
    system.selected_features = selected_features

    x_d = x[selected_features]

    return x_d

def spearman(x: pd.DataFrame, y:  pd.Series, system: SystemComponents):

    cor_target = abs(x.corrwith(y, method="spearman"))
    cor_target = cor_target.sort_values()

    features = cor_target[len(cor_target)-dimension:]

    selected_features = features.index.values
    system.selected_features = selected_features

    x_d = x[selected_features]


    return x_d

def mutual_info(x: pd.DataFrame or np.ndarray, y, system: SystemComponents):

    mi_target = mutual_info_regression(x, y)

    mi = pd.Series(mi_target, index=x.columns.values)
    mi = mi.sort_values()



    features = mi[len(mi)-dimension:]

    selected_features = features.index.values
    system.selected_features = selected_features

    x_d = x[selected_features]

    return x_d







def check_identical(f1, f2):
    identical = True
    differences = []
    for f in f1:
        if f not in f2:
            identical = False
            differences.append(f)

    return identical, differences



