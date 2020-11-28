from sklearn.linear_model import LinearRegression
from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, train_test_split
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results
from System import *


# Linear regression with ordinary least squares to diminish errors

system = SystemComponents()

if run == 'basic':
    for data in system.input_list:
        system.feature_space = data
        for proc in system.processor_list:
            system.processor = proc
            model_name = 'LR'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)

                x_transformed = preprocess(x, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                model = LinearRegression(n_jobs=-1)
                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                train_pred = pd.Series(train_pred, index=train_index, name='pred')
                test_pred = model.predict(x_test)
                test_pred = pd.Series(test_pred, index=test_index, name='pred')

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
            model_name = 'LR'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)

                x_transformed = preprocess(x, system)

                x_transformed = select_features(x_transformed, y, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                model = LinearRegression(n_jobs=-1)
                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
                test_pred = model.predict(x_test)
                test_pred = pd.Series(test_pred, index=x_test.index, name='pred')

                results = format_results(df, train_pred, test_pred, include_pred_errors)

                forecast_metrics = metrics_list(results.loc[x_test.index.values], include_pred_errors)
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

    system.feature_space = 'C'

    system.feature_engineering = ''

    system.processor = ''

    #################################################################

    model_name = 'LR'
    model_name = get_model_name(model_name, system)
    for t in market_etfs:
        x, y, df = get_data(t, system)

        x_transformed = preprocess(x, system)

        if system.feature_engineering is not None:
            x_transformed = select_features(x_transformed, y, system)

        x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

        model = LinearRegression(n_jobs=-1)
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