from System import *
from sklearn import metrics
import math


def format_results(df, train_pred: pd.Series, test_pred: pd.Series, include_pred_errors):

    train_pred = train_pred.sort_index()
    test_pred = test_pred.sort_index()

    train = pd.Series(df['Close_Forecast'].loc[train_pred.index.values], index=train_pred.index)
    test = pd.Series(df['Close_Forecast'].loc[test_pred.index.values], index=test_pred.index)

    predictions = train_pred.append(test_pred)
    predictions = predictions.sort_index()

    results = pd.DataFrame(np.nan, index=df.index.values[:-forecast_out], columns=['Train Target', 'Pred', 'Test Target'])

    results['Pred'] = predictions
    results['Train Target'] = train
    results['Test Target'] = test
    results['Close_Forecast'] = df['Close_Forecast']

    if include_pred_errors:
        test_results = get_pred_validations(results)
        results['pred_True'] = test_results['pred_True']
        results['pred_False'] = test_results['pred_False']

    results.name = df.name

    return results


def get_pred_validations(results: pd.DataFrame):
    results['pred_True'] = np.full(len(results['Pred']), np.nan, dtype=np.float)
    results['pred_False'] = np.full(len(results['Pred']), np.nan, dtype=np.float)

    for i in range(forecast_out, results.shape[0]):

        if not pd.isna(results['Test Target'].iloc[i]):

            close_pred = results['Pred'].iloc[i]
            close_true = results['Test Target'].iloc[i]

            close_last = results['Close_Forecast'].iloc[i-forecast_out]

            close_chg = (close_true - close_last) / close_last
            pred_chg = (close_pred - close_last) / close_last

            pred_sign = math.copysign(1, pred_chg)
            close_sign = math.copysign(1, close_chg)

            if (pred_sign * close_sign == 1):
                results['pred_True'][i] = close_pred
            else:
                results['pred_False'][i] = close_pred

    return results


def mean_absolute_percentage_error(actual, pred):
    # actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def get_pred_pct(pred):
    num = pred.count()
    length = len(pred)
    pct = round((num / length) * 100, 2)
    pct_str = str(pct) + '%'
    return pct_str

def metrics_list(test_results: pd.DataFrame, include_pred_errors: bool):
    y_test = test_results['Test Target']
    y_pred = test_results['Pred']


    forecast_metrics = [metrics.max_error(y_test, y_pred), metrics.mean_absolute_error(y_test, y_pred),
                        mean_absolute_percentage_error(y_test, y_pred),
                        np.sqrt(metrics.mean_squared_error(y_test, y_pred))]

    if include_pred_errors:
        pred_pct = get_pred_pct(test_results['pred_True'])
        forecast_metrics.append(pred_pct)
    return forecast_metrics

def metrics_dict(y_test, y_pred):
    forecast_metrics = {"Maximum Error (ME):": metrics.max_error(y_test, y_pred),
                        "Mean Absolute Error (MAE):": metrics.mean_absolute_error(y_test, y_pred),
                        "Mean Absolute Percentage Error (MAPE):": mean_absolute_percentage_error(y_test, y_pred),
                        "Root Mean Squared Error (RMSE):": np.sqrt(metrics.mean_squared_error(y_test, y_pred))}
    return forecast_metrics

