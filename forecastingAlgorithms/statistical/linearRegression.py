from sklearn.linear_model import LinearRegression
from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, difference, feature_difference
from preprocessing.feature_engineering.feature_engineering_master import select_features
from preprocessing.feature_engineering.feature_extraction import principal_component_analysis
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results
from modules.time_process import Timer
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

                train_size = x.index.get_loc('2018-02-26')

                x_train, y_train = x[:train_size], y[:train_size]
                x_test, y_test = x[train_size:], y[train_size:]
                test_index = x_test.index

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
    model_name = model_name + '_WF_diff'
    timer = Timer()
    for t in market_etfs:
        print(t)
        x, y, df = get_data(t, system)
        # x_diff = feature_difference(x).dropna()
        # y_diff = difference(y).dropna()

        x_diff = x.copy()
        y_diff = y.copy()

        train_start = x_diff.index.get_loc('2018-02-26')
        print('Samples to predict:', abs(train_start-len(x)))

        training_window = 50
        x_p = preprocess(x_diff, system)

        y_pred, y_test, test_index, pred_chg_list = [], [], [], []

        for i in range(train_start, len(x_diff)):

            x_train = x_diff.iloc[i - training_window:i]
            x_train = preprocess(x_train, system)
            # MAS = MaxAbsScaler()
            # MAS_fit = MAS.fit(x_train)
            # x_train = pd.DataFrame(MAS.transform(x_train), columns=x_train.columns, index=x_train.index)
            y_train = y_diff.iloc[i - training_window:i]


            date = x_diff.index.values[i]
            ts = pd.to_datetime(str(date))
            d = ts.strftime('%Y-%m-%d')
            test_index.append(date)

            x_test = np.array(x_diff.iloc[i]).reshape((1, x_diff.shape[1]))
            if system.fitted_processor is not None:
                x_test = system.fitted_processor.transform(x_test)

            # x_test = MAS_fit.transform(x_test)

            # model = LinearRegression(n_jobs=-1)
            # model.fit(x_train, y_train)
            # pred_chg = model.predict(x_test)
            # pred_chg = float(pred_chg[0])
            # pred = pred_chg + float(y.iloc[i - 1])
            # pred_chg_list.append(pred_chg)
            # y_pred.append(pred)
            # y_test.append(float(y.iloc[i]))
            # print(d, ':', round(y.iloc[i], 3), '--', round(pred,3), '    ', round(pred_chg, 2))

            model = LinearRegression(n_jobs=-1)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred = float(pred[0])
            pred_chg = pred - float(y.iloc[i - 1])
            pred_chg_list.append(pred_chg)
            y_pred.append(pred)
            y_test.append(float(y.iloc[i]))

        y_pred = pd.Series(y_pred, index=test_index, name='pred')
        y_test = pd.Series(y_test, index=test_index, name='y_test')

        results = format_results(df, y_test, y_pred)


        pred_up = round((len([p for p in pred_chg_list if p > 0]) / len(pred_chg_list)) * 100, 2)
        pred_up_str = str(pred_up) + '%'

        forecast_metrics = metrics_list(results.loc[test_index])
        forecast_metrics.append(pred_up_str)
        # print(forecast_metrics)
        errors.loc[t] = forecast_metrics

        if t=='SPY':
            plot_results(results, model_name)

    print(model_name + '          features:', x_train.shape[1])
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)
    print('elapsed time:', timer.end_timer())




if run == 'custom1':

    #################################################################

    system.feature_space = 'C'

    system.feature_engineering = ''

    system.processor = ''

    #################################################################

    model_name = 'LR'
    model_name = get_model_name(model_name, system)
    model_name = model_name + '_WF_raw'
    timer = Timer()
    for t in market_etfs:
        print(t)
        x, y, df = get_data(t, system)

        train_start = x.index.get_loc('2018-02-26')
        # print('Samples to predict:', abs(train_start-len(x)))
        training_window = 3
        x_p = preprocess(x, system)

        y_pred, y_test, test_index, pred_chg_list = [], [], [], []

        for i in range(train_start, len(x)):

            x_train = x.iloc[i - training_window:i]
            x_train = preprocess(x_train, system)
            y_train = y.iloc[i - training_window:i]


            date = x.index.values[i]
            ts = pd.to_datetime(str(date))
            d = ts.strftime('%Y-%m-%d')
            test_index.append(date)

            x_test = np.array(x.iloc[i]).reshape((1, x.shape[1]))
            if system.fitted_processor is not None:
                x_test = system.fitted_processor.transform(x_test)



            model = LinearRegression(n_jobs=-1)
            model.fit(x_train, y_train)
            # pred = model.predict(x_test)
            # pred = float(pred[0])
            # pred_chg_list.append((pred - float(y.iloc[i-1])))


            pred = float(y.iloc[i])
            pred_chg_list.append((pred - float(y.iloc[i - 1])))

            y_pred.append(pred)
            y_test.append(float(y.iloc[i]))
            # print(d, ':', round(y.iloc[i], 3), '--', round(pred,3), '    ', round(pred_chg, 2))

        y_pred = pd.Series(y_pred, index=test_index, name='pred')
        y_test = pd.Series(y_test, index=test_index, name='y_test')

        results = format_results(df, y_test, y_pred)


        pred_up = round((len([p for p in pred_chg_list if p > 0]) / len(pred_chg_list)) * 100, 2)
        pred_up_str = str(pred_up) + '%'

        forecast_metrics = metrics_list(results.loc[test_index])
        forecast_metrics.append(pred_up_str)
        # print(forecast_metrics)
        errors.loc[t] = forecast_metrics

        if t=='SPY':
            plot_results(results, model_name)

    print(model_name + '          features:', x_train.shape[1])
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)
    print('elapsed time:', timer.end_timer())