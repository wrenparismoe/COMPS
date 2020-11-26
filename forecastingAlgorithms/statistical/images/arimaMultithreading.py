import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima.model import ARIMA
from inputData.data import get_close_df
from preprocessing.process_data import train_test_split, get_model_name, preprocess
from modules.evaluation import metrics_list, format_results
from modules.plot import plot_results
from threading import Thread
from System import *


def gridSearch(data):
    p = d = q = range(0, 6)
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

lag_obs = 2
diff = 1
window = 2
order = (lag_obs, diff, window)
# order = (0, 2, 1) # original: not good order < 50%

model_name = 'ARIMA' + str(order)
model_name = get_model_name(model_name, system)


class arimaThread(Thread):
    def __init__(self, ticker):
        Thread.__init__(self)
        self.t = ticker
        self.alive = False

    def run(self):
        self.alive = True
        print(self.t, ': 0%')
        df = get_close_df(t)
        df.name = t
        df['Close_Next'] = df['Close'].shift(-1)

        x = df.drop(['Close'], 1)[:-1]
        y = df['Close_Next'][:-1]

        x_transformed = preprocess(x, system)
        df_transformed = pd.merge(x_transformed, y, left_index=True, right_index=True)

        x_train, x_test, y_train, y_test = train_test_split(x_transformed, y)
        train_size = len(x_train)

        history = [x for x in x_train.values]

        predictions = []

        # maybe need to figure out how to get rid of the for loop - only one model creation
        for d in range(len(x_test)):
            if d == len(x_test) // 2:
                print(self.t, ': 50%')
            model = ARIMA(history, order=order)
            fit = model.fit()
            pred = fit.forecast()
            print(self.t, float(pred))
            predictions.append(float(pred))
            close_last = x_test.iloc[d]
            history.append(close_last)

        print(self.t, ': 100%')

        train_pred = np.full(train_size, np.nan, dtype=np.float)
        test_pred = np.array(predictions)

        results = format_results(df, x_train, x_test, train_pred, test_pred, include_pred_errors)

        forecast_metrics = metrics_list(results[train_size:], include_pred_errors)
        errors.loc[t] = forecast_metrics

        if create_plot:
            if t == etf_to_save:
                if save_plot:
                    plot_results(results, model_name)
            plot_results(results, model_name)

        self.alive = False

thread_list = []
for t in market_etfs:
    thread = arimaThread(t)
    thread_list.append(thread)

for thread in thread_list:
    thread.start()

for thread in thread_list:
    running = True
    while running:
        if not thread.alive:
            running = False


print(model_name)
print(errors)
errors.to_clipboard(excel=True, index=False, header=False)