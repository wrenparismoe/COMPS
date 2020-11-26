from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess
from sklearn.model_selection import train_test_split
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge


system = SystemComponents()

run = 'custom'


if run == 'custom':

    #################################################################

    system.feature_space = 'OHLCTAMV'

    system.feature_engineering = 'PCA'

    system.processor = ''

    system.distribution = None

    #################################################################

    model_name = 'RF'
    model_name = get_model_name(model_name, system)
    for t in market_etfs:
        x, df = get_data(t, system)
        df.name = t
        y = df['Close_Next'][:-1]


        """ Moved Feature extraction method to before prepocessing. Was returng nonsensible results.
                       Sources say feature extraction should occur after scaling/transforming input data"""
        x_transformed = preprocess(x, system)

        x_transformed = select_features(x_transformed, y, system)

        if not isinstance(x_transformed, pd.DataFrame):
            x_transformed = pd.DataFrame(x_transformed, index=y.index)

        x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, shuffle=True, train_size=0.75)
        train_size = len(x_train)

        model = RandomForestRegressor(n_jobs=-1)
        #model = ExtraTreesRegressor()
        #model = KNN()
        #model = DecisionTreeRegressor(criterion='friedman_mse')
        model.fit(x_train, y_train)

        train_pred = model.predict(x_train)
        train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
        test_pred = model.predict(x_test)
        test_pred = pd.Series(test_pred, index=x_test.index, name='pred')

        results = format_results(df, train_pred, test_pred, include_pred_errors)

        forecast_metrics = metrics_list(results.loc[x_test.index.values], include_pred_errors)
        errors.loc[t] = forecast_metrics

        if create_plot:
            plot_results(results, model_name)


    print(model_name + '          features:', len(x_train.columns))
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)