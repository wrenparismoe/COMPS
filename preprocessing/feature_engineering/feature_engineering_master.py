from System import *
from preprocessing.feature_engineering.feature_selection import pearson, spearman, mutual_info
from preprocessing.feature_engineering.feature_extraction import principal_component_analysis


def select_features(x: pd.DataFrame or np.ndarray, y:  pd.Series or np.ndarray, system: SystemComponents):

    if system.feature_engineering == 'Pearson':
        return pearson(x, y, system)

    elif system.feature_engineering == 'Spearman':
        return spearman(x, y, system)

    elif system.feature_engineering == 'MutualInfo': # needs x,y
        return mutual_info(x, y, system)

    elif system.feature_engineering == 'PCA':
        return principal_component_analysis(x, system)

    # elif system.feature_engineering == 'SAE':
    #     return stacked_auto_encoders(x)



