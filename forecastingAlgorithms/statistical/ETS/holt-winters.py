from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # customization of params
from inputData.data import get_close_df
from preprocessing.process_data import train_test_split, get_model_name, preprocess
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results
from System import *

system = SystemComponents()
system.feature_space = 'C'

if run == 'custom':
    model_name = 'HWES'
    model_name = get_model_name(model_name, system)
    model_name = model_name + '_WF'
    for t in market_etfs:
        print(t)
        df = get_close_df(t)
        df.name = t
        df['y'] = df['Close'].shift(-forecast_out)

        x = df['Close'][:-forecast_out]
        y = df['y'][:-forecast_out]



        # train_start = x.index.get_loc('2018-04-27')
        train_start = x.index.get_loc('2012-01-01')
        print('Samples to predict:', abs(train_start - len(x)))
        training_window = 50

        y_pred, test_index, pred_chg_list = [], [], []
        y_test = y.iloc[train_start:]

        for i in range(train_start, len(x)):
            if i == (train_start + len(x)) // 2:
                print('50%')

            x_train = x.iloc[i - training_window:i]

            y_train = y.iloc[i - training_window:i]

            x_test = x.iloc[i]
            model = ETSModel(x_train, error="add", trend="add", seasonal="add", damped_trend=False,
                             seasonal_periods=2)
            fit = model.fit(maxiter=10000, disp=0)


            date = x.index.values[i]
            ts = pd.to_datetime(str(date))
            d = ts.strftime('%Y-%m-%d')
            test_index.append(date)

            forecast = fit.forecast(forecast_out)
            pred = forecast.iat[forecast_out-1]

            pred_chg_list.append((pred - float(y.iloc[i - 1])))

            y_pred.append(pred)
            # print(d, ':', round(float(y.iloc[i]), 3), '--', round(pred, 3))

        y_pred = pd.Series(y_pred, index=test_index, name='pred')

        results = format_results(df, y_test, y_pred)

        pred_up = round((len([p for p in pred_chg_list if p > 0]) / len(pred_chg_list)) * 100, 2)
        pred_up_str = str(pred_up) + '%'

        forecast_metrics = metrics_list(results.loc[test_index])
        forecast_metrics.append(pred_up_str)
        print(forecast_metrics)
        errors.loc[t] = forecast_metrics

        # if t == 'SPY':
        #     plot_results(results, model_name)
        #     exit()

    print(model_name )
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)