from System import *

from inputData.data import get_class_data
from preprocessing.process_data import get_model_name, preprocess, train_test_split
from preprocessing.feature_engineering.feature_engineering_master import select_features
from modules.evaluation import class_metrics_list, format_results
from modules.plot import plot_class_results

from math import floor

from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel

from sklearn.model_selection import train_test_split

system = SystemComponents()



if run == 'custom':

    #################################################################

    system = SystemComponents(feature_space='OHLC', feature_engineering='', processor='MMS', distribution=None)
    system.model_type = 'class'
    #################################################################

    model_name = 'SVC'
    model_name = get_model_name(model_name, system)
    for t in market_etfs:
        print(t)
        x, y, df = get_class_data(t, system)

        x_transformed = preprocess(x, system)

        if not system.feature_engineering == '':
            x_transformed = select_features(x_transformed, y, system)


        x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, shuffle=False, train_size=0.8)

        model = SVC(kernel='rbf', gamma='auto')

        model.fit(x_train, y_train)


        y_pred = model.predict(x_test)
        y_pred = pd.Series(y_pred, index=x_test.index, name='pred')

        results = format_results(df, y_test, y_pred)

        metrics = class_metrics_list(y_test, y_pred)
        classifier_errors.loc[t] = metrics

        # if t == 'QQQ':
        #     plot_class_results(results, model_name)


    print(model_name + '          features:', len(x_train.columns))
    print(classifier_errors)
    classifier_errors.to_clipboard(excel=True, index=False, header=False)


if run == 'walk_forward':
    system = SystemComponents(feature_space='OHLCTAMV', feature_engineering='MutualInfo', processor='PT',
                              distribution='normal')
    system.model_type = 'class'
    model_name = 'SVC'
    model_name = get_model_name(model_name, system)
    for t in market_etfs:
        print(t)
        x, y, df = get_class_data(t, system)

        model = SVC(kernel=RBF(length_scale=1000), C=1)

        train_start = x.index.get_loc('2016-08-17')
        print('Samples to predict:', abs(train_start-len(x)))
        training_window = 350

        x = preprocess(x, system)

        if not system.feature_engineering == '':
            x = select_features(x, y, system)

        y_pred = []
        y_test = y.iloc[train_start:]

        for i in range(train_start, len(x)):
            # if i == (train_start + len(x)) // 2:
            #     print('50%')
            x_train = x.iloc[i-training_window:i]
            x_train = preprocess(x_train, system)
            y_train = y.iloc[i-training_window:i]

            x_test = x.iloc[i:i+1]

            date = x_test.index.values[0]
            ts = pd.to_datetime(str(date))
            d = ts.strftime('%Y-%m-%d')

            x_test = system.fitted_processor.transform(x_test)

            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            y_pred.append(pred[0])
            # print(d, ':', round(y.iat[i], 4), '--', round(pred[0],4))


        y_pred = pd.Series(y_pred, index=x.index[train_start:], name='pred')

        # results = format_results(df, y_test, y_pred)

        metrics = class_metrics_list(y_test, y_pred)
        print(metrics)
        classifier_errors.loc[t] = metrics

        # if create_plot:
        #     plot_results(results, model_name)

    print(model_name + '          features:', x_train.shape[1])
    print(classifier_errors)
    classifier_errors.to_clipboard(excel=True, index=False, header=False)