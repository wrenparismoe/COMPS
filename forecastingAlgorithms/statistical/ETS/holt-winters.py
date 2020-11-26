from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # customization of params
from inputData.data import get_close_df
from preprocessing.process_data import train_test_split, get_model_name, preprocess
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results
from System import *

system = SystemComponents()
system.feature_space = 'C'

model_name = 'HWES'
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
    # pred_indices.remove(0)
    res_indices = res_indices[:-2]
    test_indices = pd.to_datetime(res_indices, format='%Y/%m/%d')

    model = ETSModel(train_data, error="add", trend="add", seasonal="add", damped_trend=False, seasonal_periods=20)
    fit = model.fit(maxiter=10000, disp=0)
    predictions = []

    for i in pred_indices:
        if i >= len(x)-forecast_out:
            break
        pred = fit.predict(start=i, end=i+7)
        predictions.append(pred[i+7])

    train_pred = np.full(train_size, np.nan, dtype=np.float)

    train_pred = pd.Series(train_pred, index=x_train.index, name='pred')
    test_pred = pd.Series(predictions, index=test_indices, name='pred')

    results = format_results(df, train_pred, test_pred, include_pred_errors)

    forecast_metrics = metrics_list(results.loc[test_indices], include_pred_errors)
    errors.loc[t] = forecast_metrics

    if create_plot and t == 'QQQ':
        plot_results(results, model_name)

print(model_name)
print(errors)
errors.to_clipboard(excel=True, index=False, header=False)