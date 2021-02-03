from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, train_test_split
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results

from sklearn.linear_model import SGDRegressor


system = SystemComponents()


if run == 'basic':
    for data in system.input_list:
        system.feature_space = data
        for proc in system.processor_list:
            system.processor = proc
            model_name = 'LRSGD'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)

                x_transformed = preprocess(x, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                model = SGDRegressor()
                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
                y_pred = model.predict(x_test)
                y_pred = pd.Series(y_pred, index=x_test.index, name='pred')

                results = format_results(df, y_test, y_pred)

                forecast_metrics = metrics_list(results.loc[test_index])
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

if run == 'derived':
    system.feature_space = 'OHLCTAMV'
    for f_e in system.feature_engineering_list:
        system.feature_engineering = f_e
        for proc in system.processor_list:
            system.processor = proc
            model_name = 'LRSGD'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)

                x_transformed = preprocess(x, system)

                x_transformed = select_features(x_transformed, y, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                model = SGDRegressor()
                model.fit(x_train, y_train)

 
                y_pred = model.predict(x_test)
                y_pred = pd.Series(y_pred, index=x_test.index, name='pred')

                results = format_results(df, y_test, y_pred)

                forecast_metrics = metrics_list(results.loc[test_index])
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

    system.distribution = 'normal'

    #################################################################

    model_name = 'LRSGD'
    model_name = get_model_name(model_name, system)
    for t in market_etfs:
        x, y, df = get_data(t, system)

        x_transformed = preprocess(x, system)

        if system.feature_engineering is not None:
            x_transformed = select_features(x_transformed, y, system)

        x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

        model = SGDRegressor()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_pred = pd.Series(y_pred, index=x_test.index, name='pred')

        results = format_results(df, y_test, y_pred)

        forecast_metrics = metrics_list(results.loc[test_index])
        errors.loc[t] = forecast_metrics

        if create_plot and t == 'QQQ':
            plot_results(results, model_name)

    print(model_name + '          features:', len(x_train.columns))
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)

def difference(data:pd.DataFrame, interval=1):
    return data - data.shift(interval)


if run == 'walk_forward':
    system = SystemComponents(feature_space='OHLC', feature_engineering='', processor='MMS',
                              distribution=None)
    model_name = 'LRSD'
    model_name = get_model_name(model_name, system)
    model_name = model_name + '_WF'
    for t in market_etfs:
        print(t)
        x, y, df = get_data(t, system)

        x = x - x.shift(1)
        y_diff = difference(y)

        model = SGDRegressor()

        train_start = x.index.get_loc('2018-02-26')
        print('Samples to predict:', abs(train_start-len(x)))
        training_window = 400

        x_p = preprocess(x, system)

        if not system.feature_engineering == '':
            x = select_features(x_p, y_diff, system)

        y_pred = []
        y_test = y.iloc[train_start:]

        for i in range(train_start, len(x)):
            if i == (train_start + len(x)) // 2:
                print('50%')
            x_train = x.iloc[i - training_window:i]
            x_train = preprocess(x_train, system)
            y_train = y_diff.iloc[i - training_window:i]

            x_test = x.iloc[i:i + 1]
            date = x_test.index.values[0]
            ts = pd.to_datetime(str(date))
            d = ts.strftime('%Y-%m-%d')

            x_test = system.fitted_processor.transform(x_test)

            model.fit(x_train, y_train)
            pred = model.predict(x_test)[0]
            pred = pred + y.iat[i - 1]

            y_pred.append(pred)
            print(d, ':', round(y.iat[i], 4), '--', round(pred, 4))

        y_pred = pd.Series(y_pred, index=test_index, name='pred')
        y_test = pd.Series(y_test, index=test_index, name='y_test')

        results = format_results(df, y_test, y_pred)

        forecast_metrics = metrics_list(results.loc[test_index])

        print(forecast_metrics)
        errors.loc[t] = forecast_metrics

        if create_plot:
            plot_results(results, model_name)

    print(model_name + '          features:', x_train.shape[1])
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)

