import plotly.graph_objects as plt
import matplotlib.dates as mdates
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from System import *



def plot_results(results: pd.DataFrame, model_name):
    etf = results.name

    fig = plt.Figure()

    fig.add_trace(plt.Scatter(x=results.index, y=results['Close_Forecast'], mode='lines',
                              line=dict(color='grey', width=3), name='Target Price',))

    for i in range(forecast_out, results.shape[0]):
        if not pd.isna(results['Test Target'].iloc[i]):
            close_pred = results['Pred'].iloc[i]
            index_pred = results.index[i]
            close_last = results['Close_Forecast'].iloc[i-forecast_out]
            index_last = results.index[i-forecast_out]

            fig.add_trace(plt.Scatter(x=[index_last, index_pred], y=[close_last, close_pred], mode='lines',
                                     line=dict(color='orange', width=3), showlegend=False))

    fig.add_trace(plt.Scatter(x=results.index, y=results['pred_False'], mode='markers',
                             marker=dict(color='tomato', size=5.5), name='Predicted Incorrectly'))

    fig.add_trace(plt.Scatter(x=results.index, y=results['pred_True'], mode='markers',
                              marker=dict(color='forestgreen', size=5.5),
                              name='Predicted Correctly'))

    if etf == etf_to_save:
        if save_plot:
            image_name = model_name + '_' + etf + '.html'
            model_folder = model_name.split('_')[0]

            online_plot_path = r"C:\Users\wrenp\Documents\COMPS\results\plots\Online interactive plots\{}".format(model_folder)
            offline_plot_path = r"C:\Users\wrenp\Documents\COMPS\results\plots\Offline png plots\{}".format(model_folder)

            fig.write_html(online_plot_path + "\{}".format(image_name))
            image_name = model_name + '_' + etf + '.png'
            fig.write_image(offline_plot_path + "\{}".format(image_name), scale=100)


    fig.update_layout(
        title=results.name + ' | Predicted vs True Closing Price (' + model_name + ')',
        xaxis_title='Date',
        yaxis_title='Close Price $'
    )
    if show_plot:
        fig.show()

def plotClose(data):
    months = mdates.MonthLocator()  # get every year
    monthsFmt = mdates.DateFormatter('%Y')  # set year format

    fig, ax = plt.subplots()

    ax.plot(data['Dates'], data['Close'])

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)

    # Set figure title
    plt.title('Close Price Since Crash', fontsize=16)
    # Set x label
    plt.xlabel('Month', fontsize=13)
    # Set y label
    plt.ylabel('Closing Price', fontsize=13)

    # Rotate and align the x labels
    fig.autofmt_xdate()

    plt.show()

def plotTrain(x_train, y_train, y_pred):
    plt.figure(1, figsize=(16, 10))
    plt.title('Linear Regression | Price vs Time | No Shuffle')
    plt.scatter(x_train, y_train, edgecolors='b', label='Actual Price')
    plt.plot(x_train, y_pred, color='r', label='Predicted Price')
    plt.xlabel('Integer Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def plot_lag(x):
    plt.figure()
    x = pd.Series(x)
    lag_plot(x, lag=3)
    plt.title('MSFT - Lag Plot')
    plt.xlabel('P(t)')
    plt.ylabel('P(t+3)')
    plt.show()

def plot_autocorrelation(x):
    plt.figure()
    autocorrelation_plot(x)
    plt.title('MSFT - Autocorrelation Plot')
    plt.show()

