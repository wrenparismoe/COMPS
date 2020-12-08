from System import *
from sklearn import metrics
import math


def format_results(df, y_test: pd.Series, y_pred: pd.Series):

    y_pred = y_pred.sort_index()

    results = pd.DataFrame(np.nan, index=df.index, columns=['x', 'y', 'y_test', 'y_pred'])

    results['x'] = df['Close']
    results['y'] = df['y']
    results['y_pred'] = y_pred
    results['y_test'] = y_test

    # if y_pred.iloc[0] == 0 or y_pred.iloc[0] == 1:
    #     results = get_class_pred_validations(results)
    # else:
    results = get_pred_validations(results)
    results.name = df.name

    return results

def get_class_pred_validations(results):
    results['y_pred_true'] = np.full(len(results['y_pred']), np.nan, dtype=np.float)
    results['y_pred_false'] = np.full(len(results['y_pred']), np.nan, dtype=np.float)

    for i in range(results.shape[0]):
        if not pd.isna(results['y_test'].iloc[i]):

            if (results['y_pred'].iloc[i] == results['y_test'].iloc[i]):
                results['y_pred_true'][i] = results['y'].iloc[i]
            else:
                results['y_pred_false'][i] = results['y'].iloc[i]

    return results


def get_pred_validations(results: pd.DataFrame):
    results['y_pred_true'] = np.full(len(results['y_pred']), np.nan, dtype=np.float)
    results['y_pred_false'] = np.full(len(results['y_pred']), np.nan, dtype=np.float)


    for i in range(results.shape[0]):
        if not pd.isna(results['y_test'].iloc[i]):

            close_pred = results['y_pred'].iloc[i]
            close_true = results['y_test'].iloc[i]

            close_last = results['x'].iloc[i-1]

            close_chg = (close_true - close_last) / close_last
            pred_chg = (close_pred - close_last) / close_last

            pred_sign = math.copysign(1, pred_chg)
            close_sign = math.copysign(1, close_chg)

            if (pred_sign * close_sign == 1):
                results['y_pred_true'][i] = close_pred
            else:
                results['y_pred_false'][i] = close_pred

    return results





def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted

def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):

    return actual[:-seasonality]

def mae(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_error(actual, predicted)))

def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):

    return mae(actual, predicted) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))

def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

def mean_absolute_percentage_error(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

def get_pred_pct(pred):
    num = pred.count()
    length = len(pred)
    pct = round((num / length) * 100, 2)
    pct_str = str(pct) + '%'
    return pct_str

def metrics_list(test_results: pd.DataFrame):
    y_test = test_results['y_test']
    y_pred = test_results['y_pred']


    # forecast_metrics = [metrics.mean_squared_error(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred),
    #                     smape(y_test, y_pred), metrics.r2_score(y_test, y_pred) ]

    forecast_metrics = [metrics.max_error(y_test, y_pred), metrics.mean_absolute_error(y_test, y_pred),
                        mean_absolute_percentage_error(y_test, y_pred),
                        np.sqrt(metrics.mean_squared_error(y_test, y_pred))]

    if include_pred_errors:
        pred_pct = get_pred_pct(test_results['y_pred_true'])
        forecast_metrics.append(pred_pct)
    return forecast_metrics

def class_metrics_list(y_test, y_pred):

    forecast_metrics = [metrics.accuracy_score(y_test, y_pred), metrics.precision_score(y_test, y_pred),
                        metrics.recall_score(y_test, y_pred), metrics.f1_score(y_test, y_pred),
                        metrics.fbeta_score(y_test, y_pred, beta=1)]

    return forecast_metrics



def metrics_dict(y_test, y_pred):
    forecast_metrics = {"Mean Squared Error (MSE):":metrics.mean_squared_error(y_test, y_pred),
                        "Mean Absolute Percentage Error (MAPE):": mean_absolute_percentage_error(y_test, y_pred),
                        "Symmetric Mean Absolute Percentage Error (sMAPE):": smape(y_test, y_pred),
                        "r2 score": metrics.r2_score(y_test, y_pred)}
    return forecast_metrics

