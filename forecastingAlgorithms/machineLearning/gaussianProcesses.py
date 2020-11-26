from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, train_test_split
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

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

                model = GaussianProcessRegressor(kernel=Matern() + RBF(), alpha=0.01)
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
            model_name = 'GP'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)

                x_transformed = preprocess(x, system)

                x_transformed = select_features(x_transformed, y, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                model = GaussianProcessRegressor(kernel=Matern() + RBF(), alpha=0.01)
                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
                test_pred = model.predict(x_test)
                test_pred = pd.Series(test_pred, index=x_test.index, name='pred')

                results = format_results(df, train_pred, test_pred, include_pred_errors)

                forecast_metrics = metrics_list(results.loc[test_index], include_pred_errors)
                errors.loc[t] = forecast_metrics

                if create_plot and t == 'QQQ':
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

    system.distribution = None

    #################################################################

    model_name = 'GP'
    model_name = get_model_name(model_name, system)
    for t in market_etfs:
        x, y, df = get_data(t, system)

        x_transformed = preprocess(x, system)

        if system.feature_engineering is not None:
            x_transformed = select_features(x_transformed, y, system)

        x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

        model = GaussianProcessRegressor(kernel=Matern() + RBF(), alpha=0.01)
        model.fit(x_train, y_train)

        train_pred = model.predict(x_train)


        train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
        test_pred = model.predict(x_test)
        test_pred = pd.Series(test_pred, index=x_test.index, name='pred')

        results = format_results(df, train_pred, test_pred, include_pred_errors)

        forecast_metrics = metrics_list(results.loc[test_index], include_pred_errors)
        errors.loc[t] = forecast_metrics

        if create_plot and t == 'QQQ':
            plot_results(results, model_name)


    print(model_name + '          features:', len(x_train.columns))
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)











