from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, difference, invert_difference, feature_difference
from preprocessing.feature_engineering.feature_engineering_master import select_features
from preprocessing.feature_engineering.feature_extraction import principal_component_analysis
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results
from modules.time_process import Timer
from time import sleep

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel, RationalQuadratic



system = SystemComponents()



if run == 'basic':
    for data in system.input_list:
        system.feature_space = data
        for proc in system.processor_list:
            system.processor = proc
            model_name = 'SVR'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)

                x_transformed = preprocess(x, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                model = SVR(kernel=Matern() + RBF(), C=1000)
                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
                test_pred = model.predict(x_test)
                test_pred = pd.Series(test_pred, index=x_test.index, name='pred')

                results = format_results(df, train_pred, test_pred, include_pred_errors)

                forecast_metrics = metrics_list(results.loc[test_index], include_pred_errors)
                errors.loc[t] = forecast_metrics

                if create_plot and t == etf_to_save:
                    plot_results(results, model_name)


            print(model_name + '          features:', len(x_train.columns))
            print(errors)
            errors.to_clipboard(excel=True, index=False, header=False)

            print()
            cont = input('continue? - type y:  ')
            if cont == 'y':
                print()
                continue
            else:
                exit()

if run == 'derived':
    system.feature_space = 'OHLCTAMV'
    for f_e in system.feature_engineering_list:
        system.feature_engineering = f_e
        for proc in system.processor_list:
            system.processor = proc
            model_name = 'SVR'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)

                x_transformed = preprocess(x, system)

                x_transformed = select_features(x_transformed, y, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                model = SVR(kernel=Matern() + RBF(), C=1000)
                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
                test_pred = model.predict(x_test)
                test_pred = pd.Series(test_pred, index=x_test.index, name='pred')

                results = format_results(df, train_pred, test_pred, include_pred_errors)

                forecast_metrics = metrics_list(results.loc[test_index], include_pred_errors)
                errors.loc[t] = forecast_metrics

                if create_plot and t == etf_to_save:
                    plot_results(results, model_name)

            print(model_name + '          features:', len(x_train.columns))
            print(errors)
            errors.to_clipboard(excel=True, index=False, header=False)

            print()
            cont = input('continue? - type y:  ')
            if cont == 'y':
                print()
                continue
            else:
                exit()

if run == 'custom':

    #################################################################

    system.feature_space = 'OHLCTAMV'

    system.feature_engineering = 'MutualInfo'

    system.processor = 'PT'

    system.distribution = 'normal'

    #################################################################
    timer = Timer()
    model_name = 'SVR'
    model_name = get_model_name(model_name, system)
    for t in market_etfs:
        x, y, df = get_data(t, system)

        x_transformed = preprocess(x, system)

        if system.feature_engineering != '':
            x_transformed = select_features(x_transformed, y, system)

        x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

        model = SVR(kernel=Matern() + RBF(), C=1000)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_pred = pd.Series(y_pred, index=x_test.index, name='pred')

        results = format_results(df, y_test, y_pred)

        forecast_metrics = metrics_list(results.loc[test_index])
        errors.loc[t] = forecast_metrics

        if t == 'SPY':
            plot_results(results, model_name)

    print(model_name + '          features:', len(x_train.columns))
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)
    runtime = timer.end_timer()
    print('runtime:', runtime)

def gridSearch() -> SVR:

    x_train = x_diff.iloc[:train_start]
    x_train = preprocess(x_train, system)
    y_train = y_diff.iloc[:train_start]

    x_test = np.array(x_diff.iloc[train_start]).reshape((1, x_diff.shape[1]))
    if system.fitted_processor is not None:
        x_test = system.fitted_processor.transform(x_test)
    y_test = np.array(y.iloc[train_start]).reshape(1, -1)

    param_grid = {'gamma': [0.000001, 0.00001], 'C': [1, 2, 5, 10],
                  'tol': [0.0001, 0.001, 0.01], 'epsilon': [0.1, 0.001, 0.0001]}

    svr = SVR(kernel=RBF())
    GS = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1,
                      cv=math.ceil(train_start/training_window))
    GS.fit(x_train, y_train)
    
    print('Parameters:', GS.best_params_)
    print('Model:', GS.best_estimator_)
    print('Score', GS.best_score_)
    gs_running = False
    return GS.best_estimator_

def difference(data:pd.DataFrame, interval=1):
    return data - data.shift(interval)
























