from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, train_test_split
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results, format_forecast_results
from modules.plot import plot_results
# from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from preprocessing.transform_data.transformer import PowerTransformer


system = SystemComponents(feature_space='OHLCTAMV', feature_engineering='MutualInfo', processor='PT', distribution='normal')





model_name = 'GP'
model_name = get_model_name(model_name, system)
for t in market_etfs:
    print(t)
    x, y, df = get_data(t, system, True)

    x_transformed = preprocess(x, system)

    x_transformed = select_features(x_transformed, y, system)

    x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

    model = GaussianProcessRegressor(kernel=Matern() + RBF(), alpha=0.001)
    model.fit(x_train, y_train)





    forecast_results = format_forecast_results(df_f, y_pred_f)

    forecast_metrics = metrics_list(forecast_results)
    f_errors.loc[t] = forecast_metrics

    results = results.append(forecast_results)
    results.name = t

    if t == 'SPY':
        plot_results(results, model_name)


#print(model_name + '          features:', len(x_transformed.columns))
print(errors)
# print(f_errors)
f_errors.to_clipboard(excel=True, index=False, header=False)


