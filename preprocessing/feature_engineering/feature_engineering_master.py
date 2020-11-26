from System import *
from preprocessing.feature_engineering.feature_selection import pearson, mutual_info
from preprocessing.feature_engineering.feature_extraction import principal_component_analysis
from preprocessing.feature_engineering.feature_extraction import stacked_auto_encoders

def select_features(x: pd.DataFrame or np.ndarray, y:  pd.Series or np.ndarray, system: SystemComponents):
    x_d = x
    if system.feature_engineering == 'Pearson':
        return pearson(x, y, dimension)

    elif system.feature_engineering == 'MutualInfo': # needs x,y
        return mutual_info(x, y, dimension)

    elif system.feature_engineering == 'PCA':
        return principal_component_analysis(x)

    elif system.feature_engineering == 'SAE':
        return stacked_auto_encoders(x)

    return x_d

