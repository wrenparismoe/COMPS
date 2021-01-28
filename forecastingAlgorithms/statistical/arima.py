import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.api import ARIMA
from inputData.data import get_close_df, get_data
from preprocessing.process_data import train_test_split, get_model_name, preprocess
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results
from datetime import datetime

from System import *


def gridSearch(data):
    p = d = q = range(0, 7)
    pdq = list(itertools.product(p, d, q))
    # all parameter combinations
    aic = []
    parameters = []
    for param in pdq:
        # for param in pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data, order=param,
                                            enforce_stationarity=True, enforce_invertibility=True)
            results = mod.fit()
            # save results in lists
            aic.append(results.aic)
            parameters.append(param)
            # seasonal_param.append(param_seasonal)
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
        except:
            continue
    # find lowest aic
    index_min = min(range(len(aic)), key=aic.__getitem__)
    print('The optimal model is: ARIMA{} -AIC{}'.format(parameters[index_min], aic[index_min]))



system = SystemComponents()
system.feature_space = 'C'
system.processor = ''

lag_obs = 2
diff = 5
window = 5
order = (lag_obs, diff, window)
# order = (0, 2, 1) # original: not good order < 50%


if run == 'walk_forward1':
    model_name = 'ARIMA'
    model_name = get_model_name(model_name, system)

    for t in market_etfs:
        df = get_close_df(t)
        df.name = t
        df['Close_Forecast'] = df['Close'].shift(-forecast_out)

        x = df['Close'][:-forecast_out]
        y = df['Close_Forecast'][:-forecast_out]

        x_train, x_test, y_train, y_test= train_test_split(x, y)
        train_size = len(x_train)

        x_train: pd.Series = x_train.sort_index()
        x_test: pd.Series = x_test.sort_index()

        train_data = pd.Series(x.values, index=[x for x in range(len(x))])

        ind = y.index.values
        ind = pd.to_datetime(ind).strftime('%Y-%m-%d')
        indices = []
        for ind in ind:
            indices.append(ind)

        test_ind = y_test.index.values
        test_ind = pd.to_datetime(test_ind).strftime('%Y-%m-%d')
        y_test_indices = []
        for ind in test_ind:
            y_test_indices.append(ind)

        pred_indices = []
        res_indices = []
        for i in range(len(indices)):
            ind_curr = indices[i]
            if ind_curr in y_test_indices:
                pred_indices.append(i)
                res_indices.append(ind_curr)
        pred_indices.remove(0)
        res_indices = res_indices[1:-2]
        test_indices = pd.to_datetime(res_indices, format='%Y/%m/%d')


        model = ARIMA(endog=train_data, order=order)
        fit = model.fit()
        predictions = []

        for i in pred_indices:
            if i >= len(x)-forecast_out:
                break
            pred = fit.predict(start=i, end=i+7, typ='levels')
            predictions.append(pred[i+7])

        train_pred = np.full(train_size, np.nan, dtype=np.float)

        train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
        test_pred = pd.Series(predictions, index=test_indices, name='pred')

        results = format_results(df, train_pred, test_pred, include_pred_errors)

        forecast_metrics = metrics_list(results.loc[test_indices], include_pred_errors)
        errors.loc[t] = forecast_metrics

        if create_plot and t == 'DIA':
            plot_results(results, model_name)

    print(model_name)
    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)

lag_obs = 3
diff = 1
window = 4
order = (lag_obs, diff, window)

if run == 'custom':
    model_name = 'ARIMA'
    model_name = get_model_name(model_name, system)
    model_name = model_name + '_WF'
    for t in market_etfs:
        print(t)

        x, y, df = get_data(t, system)
        x = df['Close'][:-1]

        train_start = x.index.get_loc('2018-02-26')
        print('Samples to predict:', abs(train_start - len(x)))
        training_window = 75

        y_pred, y_test, test_index, pred_chg_list = [], [], [], []



        for i in range(train_start, len(x)-1):
            if i == (train_start + len(x)) // 2:
                print('50%')

            x_train = x.iloc[i - training_window:i]

            x_test = x.iloc[i]
            model = ARIMA(endog=x_train, order=order)
            fit = model.fit()

            date = x.index.values[i]
            ts = pd.to_datetime(str(date))
            d = ts.strftime('%Y-%m-%d')
            test_index.append(date)

            forecast = fit.forecast(1)
            # print(forecast)
            pred = forecast.astype(np.float64).values[0]

            pred_chg_list.append((pred - float(y.iloc[i-1])))
            y_test.append(float(y.iloc[i]))
            y_pred.append(pred)
            # print(d, ':', round(float(y.iloc[i]), 3), '--', round(pred,3))

        y_pred = pd.Series(y_pred, index=test_index, name='pred')
        y_test = pd.Series(y_test, index=test_index, name='y_test')
        results = format_results(df, y_test, y_pred)

        pred_up = round((len([p for p in pred_chg_list if p > 0]) / len(pred_chg_list)) * 100, 2)
        pred_up_str = str(pred_up) + '%'

        forecast_metrics = metrics_list(results.loc[test_index])
        forecast_metrics.append(pred_up_str)
        print(forecast_metrics)
        errors.loc[t] = forecast_metrics

        if t =='SPY':
            plot_results(results, model_name)






    print(errors)
    errors.to_clipboard(excel=True, index=False, header=False)