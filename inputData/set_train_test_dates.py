from inputData.data import get_data
from sklearn.model_selection import train_test_split
from preprocessing.process_data import preprocess
from System import *


system = SystemComponents()

system.feature_space = 'C'

x, y, df = get_data('SPY', system)

x_transformed = preprocess(x, system)


x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, shuffle=True, train_size=0.75)

# print(type(x_train))
# print(type(x_test))

if isinstance(x_train, pd.DataFrame) and isinstance(x_test, pd.DataFrame):
    print()

    x_train.to_csv(r'dataFiles\train_index.csv', columns=[], index=True, header=True)
    x_test.to_csv(r'dataFiles\test_index.csv', columns=[], index=True, header=True)


