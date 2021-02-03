import numpy as np
import pandas as pd

from technicalAnalysis.utils import IndicatorMixin


class DailyReturnIndicator(IndicatorMixin):


    def __init__(self, close: pd.Series, fillna: bool = False):
        self._close = close
        self._fillna = fillna
        self._run()

    def _run(self):
        self._dr = (self._close / self._close.shift(1, fill_value=self._close.mean())) - 1
        self._dr *= 100

    def daily_return(self) -> pd.Series:

        dr = self._check_fillna(self._dr, value=0)
        return pd.Series(dr, name='d_ret')


class DailyLogReturnIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, fillna: bool = False):
        self._close = close
        self._fillna = fillna
        self._run()

    def _run(self):
        self._dr = np.log(self._close).diff()
        self._dr *= 100

    def daily_log_return(self) -> pd.Series:

        dr = self._check_fillna(self._dr, value=0)
        return pd.Series(dr, name='d_logret')


class CumulativeReturnIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, fillna: bool = False):
        self._close = close
        self._fillna = fillna
        self._run()

    def _run(self):
        self._cr = (self._close / self._close.iloc[0]) - 1
        self._cr *= 100

    def cumulative_return(self) -> pd.Series:
        cr = self._check_fillna(self._cr, value=-1)
        return pd.Series(cr, name='cum_ret')


def daily_return(close, fillna=False):
    return DailyReturnIndicator(close=close, fillna=fillna).daily_return()


def daily_log_return(close, fillna=False):
    return DailyLogReturnIndicator(close=close, fillna=fillna).daily_log_return()


def cumulative_return(close, fillna=False):
    return CumulativeReturnIndicator(close=close, fillna=fillna).cumulative_return()