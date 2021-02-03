from System import *

from inputData.data import get_data
from preprocessing.process_data import get_model_name, preprocess, difference, invert_difference, feature_difference
from preprocessing.feature_engineering.feature_engineering_master import select_features
from preprocessing.feature_engineering.feature_extraction import principal_component_analysis
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results
from modules.time_process import Timer

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from scipy import stats


system = SystemComponents()

if run == 'basic':
    for data in system.input_list:
        system.feature_space = data
        for proc in system.processor_list:
            system.processor = proc
            model_name = 'GP'
            model_name = get_model_name(model_name, system)
            timer = Timer()
            for t in market_etfs:
                print(t)
                x, y, df = get_data(t, system)
                x_diff = feature_difference(x).dropna()
                y_diff = difference(y).dropna()

                train_start = x_diff.index.get_loc('2018-02-26')
                print('Samples to predict:', abs(train_start-len(x)))
                training_window = 75

                y_pred, y_test, test_index, pred_chg_list = [], [], [], []

                for i in range(train_start, len(x_diff)):
                    # if i == (train_start + len(x_diff)) // 2:
                    #     print('50%')
                    x_train = x_diff.iloc[i - training_window:i]
                    x_train = preprocess(x_train, system)
                    y_train = y_diff.iloc[i - training_window:i]

                    date = x_diff.index[i]
                    ts = pd.to_datetime(str(date))
                    d = ts.strftime('%Y-%m-%d')
                    test_index.append(date)

                    x_test = np.array(x_diff.iloc[i]).reshape((1, x_diff.shape[1]))
                    if system.fitted_processor is not None:
                        x_test = system.fitted_processor.transform(x_test)


                    kernel = RBF() + WhiteKernel(1)
                    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
                    model.fit(x_train, y_train)
                    pred_chg = model.predict(x_test)
                    pred = pred_chg[0] + y.iloc[i - 1]
                    pred_chg_list.append(pred_chg[0])
                    y_pred.append(pred)
                    y_test.append(y.iloc[i])
                    # print(d, ':', round(y.iloc[i], 3), '--', round(pred,3), '    ', round(pred_chg[0], 2))

                y_pred = pd.Series(y_pred, index=test_index, name='pred')
                y_test = pd.Series(y_test, index=test_index, name='y_test')

                results = format_results(df, y_test, y_pred)

                pred_up = round((len([p for p in pred_chg_list if p > 0]) / len(pred_chg_list)) * 100, 2)
                pred_up_str = str(pred_up) + '%'

                forecast_metrics = metrics_list(results.loc[test_index])
                forecast_metrics.append(pred_up_str)
                #print(forecast_metrics)
                errors.loc[t] = forecast_metrics

                plot_results(results, model_name)

            print(model_name + '          features:', x_train.shape[1])
            print(errors)
            errors.to_clipboard(excel=True, index=False, header=False)
            print('elapsed time:', timer.end_timer())

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
            timer = Timer()
            for t in market_etfs:
                print(t)
                x, y, df = get_data(t, system)

                x_diff = feature_difference(x).dropna()
                y_diff = difference(y).dropna()

                train_start = x_diff.index.get_loc('2018-02-26')
                print('Samples to predict:', abs(train_start-len(x)))
                training_window = 65

                x_p = preprocess(x_diff, system)

                if not system.feature_engineering == '' and not system.feature_engineering == 'PCA':
                    x_proc = select_features(x_p, y_diff, system)
                    x_diff = x_diff[system.selected_features]
                    print('features:', x_proc.shape[1])

                y_pred, y_test, test_index, pred_chg_list = [], [], [], []

                for i in range(train_start, len(x_diff)):
                    # if i == (train_start + len(x_diff)) // 2:
                    #     print('50%')
                    x_train = x_diff.iloc[i - training_window:i]
                    x_train = preprocess(x_train, system)
                    y_train = y_diff.iloc[i - training_window:i]

                    if system.feature_engineering == 'PCA':
                        x_train = principal_component_analysis(x_train, system)

                    date = x_diff.index[i]
                    ts = pd.to_datetime(str(date))
                    d = ts.strftime('%Y-%m-%d')
                    test_index.append(date)

                    x_test = np.array(x_diff.iloc[i]).reshape((1, x_diff.shape[1]))
                    if system.fitted_processor is not None:
                        x_test = system.fitted_processor.transform(x_test)
                    if system.feature_engineering == 'PCA':
                        x_test = system.fitted_pca.transform(x_test)

                    var_ = np.mean(stats.norm.pdf(x_train), axis=1)

                    kernel = RBF()
                    model = GaussianProcessRegressor(kernel=kernel, alpha=var_, normalize_y=True)
                    model.fit(x_train, y_train)
                    pred_chg = model.predict(x_test)
                    pred = pred_chg[0] + y.iloc[i - 1]
                    pred_chg_list.append(pred_chg[0])
                    y_pred.append(pred)
                    y_test.append(y.iloc[i])
                    # print(d, ':', round(y.iloc[i], 3), '--', round(pred,3), '    ', round(pred_chg[0], 2))

                y_pred = pd.Series(y_pred, index=test_index, name='pred')
                y_test = pd.Series(y_test, index=test_index, name='y_test')

                results = format_results(df, y_test, y_pred)

                pred_up = round((len([p for p in pred_chg_list if p > 0]) / len(pred_chg_list)) * 100, 2)
                pred_up_str = str(pred_up) + '%'

                forecast_metrics = metrics_list(results.loc[test_index])
                forecast_metrics.append(pred_up_str)
                #print(forecast_metrics)
                errors.loc[t] = forecast_metrics

                if create_plot:
                    plot_results(results, model_name)

            print(model_name + '          features:', x_train.shape[1])
            print(errors)
            errors.to_clipboard(excel=True, index=False, header=False)
            print(timer.end_timer())

            print()
            cont = input('continue? - type y:  ')
            if cont == 'y':
                print()
                continue
            else:
                exit()



if run == 'custom':
    timer = Timer()
    system = SystemComponents(feature_space='OHLCTAMV', feature_engineering='MutualInfo', processor='SS')
    model_name = 'GP'
    model_name = get_model_name(model_name, system)
    model_name = model_name + '_WF'
    for t in market_etfs:
        print(t)
        x, y, df = get_data(t, system)
        x_diff = feature_difference(x).dropna()
        y_diff = difference(y).dropna()


        train_start = x_diff.index.get_loc('2018-02-26')

        training_window = 100

        y_pred, y_test, test_index, pred_chg_list, pred_yes_list = [], [], [], [], []
        for i in range(train_start, len(x_diff)):

            x_train = x_diff.iloc[i - training_window:i]
            x_train = preprocess(x_train, system)
            MAS = MaxAbsScaler()
            MAS_fit = MAS.fit(x_train)
            x_train = pd.DataFrame(MAS.transform(x_train), columns=x_train.columns, index=x_train.index)
            y_train = y_diff.iloc[i - training_window:i]

            if not system.feature_engineering == '' and not system.feature_engineering == 'PCA':
                x_train = select_features(x_train, y_train, system)
                x_train = x_train[system.selected_features]
                print('features:', x_train.shape[1])
            elif system.feature_engineering == 'PCA':
                x_train = principal_component_analysis(x_train, system)

            date = x_diff.index[i]
            ts = pd.to_datetime(str(date))
            d = ts.strftime('%Y-%m-%d')
            test_index.append(date)

            x_test = np.array(x_diff.iloc[i]).reshape((1, x_diff.shape[1]))
            if system.fitted_processor is not None:
                x_test = system.fitted_processor.transform(x_test)
            x_test = MAS_fit.transform(x_test)
            x_test = pd.DataFrame(x_test, columns=x_diff.columns, index=x_diff.index[i:i+1])

            if not system.feature_engineering == '' and not system.feature_engineering == 'PCA':
                x_test = x_test[system.selected_features]

            if system.feature_engineering == 'PCA':
                x_test = system.fitted_pca.transform(x_test)

            kernel = RBF() + WhiteKernel(noise_level=1)
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            model.fit(x_train, y_train)
            pred_chg= model.predict(x_test)
            pred = pred_chg[0] + y.iloc[i - 1]

            pred_chg_list.append(pred_chg)
            y_pred.append(pred)

            y_test.append(y.iloc[i])
            #print(d, ':', round(y.iloc[i], 3), '--', round(pred,3), '    ', round(pred_chg[0], 2))

        y_pred = pd.Series(y_pred, index=test_index, name='pred')
        y_test = pd.Series(y_test, index=test_index, name='y_test')

        results = format_results(df, y_test, y_pred)

        pred_up = round((len([p for p in pred_chg_list if p > 0]) / len(pred_chg_list)) * 100, 2)
        pred_up_str = str(pred_up) + '%'

        forecast_metrics = metrics_list(results.loc[test_index])
        forecast_metrics.append(pred_up_str)
        print(forecast_metrics)
        errors.loc[t] = forecast_metrics

        if t == 'SPY':
            plot_results(results, model_name)

    print(model_name + '          features:', x_train.shape[1])
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)
    print(timer.end_timer())