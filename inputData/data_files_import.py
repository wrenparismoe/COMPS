import pandas as pd

SPY_C = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/closeData/SPY_C.csv')
SPY_C.name = 'SPY_C'
QQQ_C = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/closeData/QQQ_C.csv')
QQQ_C.name = 'QQQ_C'
IWM_C = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/closeData/IWM_C.csv')
IWM_C.name = 'IWM_C'
DIA_C = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/closeData/DIA_C.csv')
DIA_C.name = 'DIA_C'

SPY_OHLC = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/ohlcData/SPY_OHLC.csv')
SPY_OHLC.name = 'SPY_OHLC'
QQQ_OHLC = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/ohlcData/QQQ_OHLC.csv')
QQQ_OHLC.name = 'QQQ_OHLC'
IWM_OHLC = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/ohlcData/IWM_OHLC.csv')
IWM_OHLC.name = 'IWM_OHLC'
DIA_OHLC = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/ohlcData/DIA_OHLC.csv')
DIA_OHLC.name = 'DIA_OHLC'

SPY_OHLCTAMV = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/ohlctamvData/SPY_OHLCTAMV.csv')
SPY_OHLCTAMV.name = 'SPY_OHLCTAMV'
QQQ_OHLCTAMV = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/ohlctamvData/QQQ_OHLCTAMV.csv')
QQQ_OHLCTAMV.name = 'QQQ_OHLCTAMV'
IWM_OHLCTAMV = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/ohlctamvData/IWM_OHLCTAMV.csv')
IWM_OHLCTAMV.name = 'IWM_OHLCTAMV'
DIA_OHLCTAMV = pd.read_csv('C:/Users/wrenp/COMPS Workspace/inputData/dataFiles/ohlctamvData/DIA_OHLCTAMV.csv')
DIA_OHLCTAMV.name = 'DIA_OHLCTAMV'


Close_Data = [SPY_C, QQQ_C, IWM_C, DIA_C]
OHLC_Data = [SPY_OHLC, QQQ_OHLC, IWM_OHLC, DIA_OHLC]
OHLCTAMV_Data = [SPY_OHLCTAMV, QQQ_OHLCTAMV, IWM_OHLCTAMV, DIA_OHLCTAMV]

Multivariate_Data = [SPY_OHLC, QQQ_OHLC, IWM_OHLC, DIA_OHLC, SPY_OHLCTAMV, QQQ_OHLCTAMV, IWM_OHLCTAMV,DIA_OHLCTAMV]

All_Data = [SPY_C, QQQ_C, IWM_C, DIA_C, SPY_OHLC, QQQ_OHLC, IWM_OHLC, DIA_OHLC, SPY_OHLCTAMV, QQQ_OHLCTAMV, IWM_OHLCTAMV,DIA_OHLCTAMV]