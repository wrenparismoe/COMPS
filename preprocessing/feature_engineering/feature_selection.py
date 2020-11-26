from sklearn.feature_selection import SelectKBest, mutual_info_regression
from System import *



def pearson(x: pd.DataFrame or np.ndarray, y:  pd.Series or np.ndarray, dim: int = dimension):
    if isinstance(x, pd.DataFrame):
        cor_target = abs(x.corrwith(y))
        cor_target = cor_target.sort_values()

        # threshold_features = cor_target[cor_target > 0.9]

        relevant_features = cor_target[len(cor_target)-(dim+1):-1]
        selected_features = relevant_features.index.values
        x = x[selected_features]

    # elif isinstance(x, np.ndarray):
    #     cor_target = np.corrcoef(x, y, rowvar=False)[len(x[0]),:-1]
    #     cor_target = abs(cor_target)
    #     ind = np.argpartition(cor_target, -dim)[-dim:]
    #     ind = np.sort(ind)
    #     selected_features = np.array(x)[:,ind.astype(int)]


    return x



def mutual_info(x: pd.DataFrame or np.ndarray, y, dim: int = dimension):
    mi_target = mutual_info_regression(x, y)

    mi_df = pd.Series(data=mi_target, index=x.columns.values)
    mi_df = mi_df.sort_values()[len(mi_df)-dim:]

    selected_features = mi_df.index.values

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



