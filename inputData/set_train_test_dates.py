from inputData.data import get_data
from sklearn.model_selection import train_test_split

from System import *


system = SystemComponents()

system.feature_space = 'C'

x, y, df = get_data('SPY', system)



x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.80)



if isinstance(x_train, pd.DataFrame) and isinstance(x_test, pd.DataFrame):

    x_train.to_csv(r'dataFiles\train_index.csv', columns=[], index=True, header=True)
    x_test.to_csv(r'dataFiles\test_index.csv', columns=[], index=True, header=True)
    print('done')


