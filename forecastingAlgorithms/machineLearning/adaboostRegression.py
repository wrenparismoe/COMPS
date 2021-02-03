from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, difference, invert_difference, feature_difference, train_test_split
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

system = SystemComponents()


if run == 'basic':
    for data in system.input_list:
        system.feature_space = data
        for proc in system.processor_list:
            system.processor = proc
            model_name = 'DT'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)

                x_transformed = preprocess(x, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                DT = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3, min_samples_split=2)
                model = AdaBoostRegressor(base_estimator=DT, n_estimators=50, learning_rate=0.75, loss='linear')

                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
                test_pred = model.predict(x_test)
                test_pred = pd.Series(test_pred, index=x_test.index, name='pred')

                results = format_results(df, train_pred, test_pred)

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
            model_name = 'DT'
            model_name = get_model_name(model_name, system)
            for t in market_etfs:
                x, y, df = get_data(t, system)

                x_transformed = preprocess(x, system)

                x_transformed = select_features(x_transformed, y, system)

                x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

                DT = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3, min_samples_split=2)
                model = AdaBoostRegressor(base_estimator=DT, n_estimators=50, learning_rate=0.75, loss='linear')
                model.fit(x_train, y_train)

                train_pred = model.predict(x_train)
                train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
                test_pred = model.predict(x_test)
                test_pred = pd.Series(test_pred, index=x_test.index, name='pred')

                results = format_results(df, train_pred, test_pred)

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

    system = SystemComponents(feature_space = 'OHLCTAMV', feature_engineering = 'MutualInfo', processor = 'PT')

    #################################################################

    model_name = 'DT'
    model_name = get_model_name(model_name, system)
    for t in market_etfs:
        x, y, df = get_data(t, system)

        x_transformed = preprocess(x, system)

        if not system.feature_engineering == '':
            x_transformed = select_features(x_transformed, y, system)

        x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)

        DT = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3, min_samples_split=2)
        model = AdaBoostRegressor(base_estimator=DT, n_estimators=50, learning_rate=0.75, loss='linear')
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


def difference(data:pd.DataFrame, interval=1):
    return data - data.shift(interval)




