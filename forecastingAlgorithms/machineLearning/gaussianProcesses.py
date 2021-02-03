from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, difference, invert_difference, feature_difference
from preprocessing.feature_engineering.feature_engineering_master import select_features
from preprocessing.feature_engineering.feature_extraction import principal_component_analysis
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results
from modules.time_process import Timer

from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
system = SystemComponents()




if run == 'basic':
    for data in system.input_list:
        system.feature_space = data
        for proc in system.processor_list:
            system.processor = proc
            model_name = 'GP'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)


                x_transformed = preprocess(x, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                kernel = RBF() + WhiteKernel(1)
                model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
                model.fit(x_train, y_train)

                y_pred = model.predict(x_test)
                y_pred = pd.Series(y_pred, index=x_test.index, name='pred')

                results = format_results(df, y_test, y_pred)

                forecast_metrics = metrics_list(results.loc[test_index])
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
            model_name = 'GP'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)

                x_transformed = preprocess(x, system)

                x_transformed = select_features(x_transformed, y, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                kernel = RBF() + WhiteKernel(1)
                model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
                y_pred = model.predict(x_test)
                y_pred = pd.Series(y_pred, index=x_test.index, name='pred')

                results = format_results(df, train_pred, y_pred)

                forecast_metrics = metrics_list(results.loc[test_index])
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

    system = SystemComponents(feature_space='OHLCTAMV', feature_engineering='Pearson', processor='PT', distribution=None)

    #################################################################

    model_name = 'GP'
    model_name = get_model_name(model_name, system)
    for t in market_etfs:
        print(t)
        x, y, df = get_data(t, system)

        x_diff = feature_difference(x).dropna()
        y_diff = difference(y).dropna()

        x_p = preprocess(x, system)

        if not system.feature_engineering == '':
            x = select_features(x_p, y_diff, system)


        train_size = math.floor(len(x) * 0.75)
        x_train, y_train = x[:train_size], y_diff[:train_size]
        x_test, y_test = x[train_size:], y[train_size:]
        test_index = x_test.index

        x_train = preprocess(x_train, system)
        x_test = system.fitted_processor.transform(x_test)

        kernel = RBF() + WhiteKernel(1)
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        model.fit(x_train, y_train)


        y_pred = model.predict(x_test)
        y_pred = pd.Series(y_pred, index=test_index, name='pred')
        y_pred = invert_difference(y_test, y_pred)

        results = format_results(df, y_test, y_pred)


        forecast_metrics = metrics_list(results.loc[test_index[1:-1]])
        errors.loc[t] = forecast_metrics

        if t == 'SPY':
            plot_results(results, model_name)


    print(model_name + '          features:', len(x_train.columns))
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)










