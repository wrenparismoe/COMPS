from plotly import graph_objs as plt
import matplotlib.dates as mdates
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
import chart_studio
import chart_studio.plotly as py
from System import *

chart_studio.tools.set_credentials_file(**studio_params)


def plot_class_results(results: pd.DataFrame, model_name):
    etf = results.name

    #results = results.iloc[783:]

    fig = plt.Figure()

    fig.add_trace(plt.Scatter(x=results.index, y=results['y'], mode='lines',
                              line=dict(color='grey', width=3), name='Target Price', ))


    fig.add_trace(plt.Scatter(x=results.index, y=results['y_pred_false'], mode='markers',
                             marker=dict(color='tomato', size=5.5), name='Incorrect Pred'))



    fig.add_trace(plt.Scatter(x=results.index, y=results['y_pred_true'], mode='markers',
                              marker=dict(color='forestgreen', size=5.5),
                              name='Correct Pred'))

    fig.update_layout(
        title=etf + ' | Predicted vs True Closing Price (' + model_name + ')', xaxis_title='Date',
        yaxis_title='Close Price $', font=dict(size=15)
    )
    fig.update_layout(legend_font_size=15)
    fig.update_layout(legend_font_size=15)

    if etf == etf_to_save:
        if save_plot:
            file_name = model_name + '_' + etf

            py.plot(fig, filename=file_name, auto_open=show_plot)

            return

    if show_plot:
        fig.show()





def plot_results(results: pd.DataFrame, model_name):
    etf = results.name

    # results = results.iloc[783:]

    fig = plt.Figure()

    for i in range(results.shape[0]):
        if not pd.isna(results['y_test'].iloc[i]):
            close_pred = results['y_pred'].iloc[i]
            index_pred = results.index[i]
            close_last = results['x'].iloc[i-1]
            index_last = results.index[i-1]

            fig.add_trace(plt.Scatter(x=[index_last, index_pred], y=[close_last, close_pred], mode='lines',
                                     line=dict(color='orange', width=2), showlegend=False))


    fig.add_trace(plt.Scatter(x=results.index, y=results['x'], mode='lines',
                              line=dict(color='grey', width=3), name='Target Price', ))

    fig.add_trace(plt.Scatter(x=results.index, y=results['y_pred_false'], mode='markers',
                             marker=dict(color='tomato', size=5.5), name='Incorrect Pred'))



    fig.add_trace(plt.Scatter(x=results.index, y=results['y_pred_true'], mode='markers',
                              marker=dict(color='forestgreen', size=5.5),
                              name='Correct Pred'))



    fig.update_layout(
        title=etf + ' | Predicted vs True Closing Price (' + model_name + ')', xaxis_title='Date',
        yaxis_title='Close Price $', font=dict(size=15)
    )
    fig.update_layout(legend_font_size=15)
    fig.update_layout(legend_font_size=15)

    if etf == etf_to_save:
        if save_plot:
            file_name = model_name + '_' + etf

            py.plot(fig, filename=file_name, auto_open=show_plot)

            return

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

