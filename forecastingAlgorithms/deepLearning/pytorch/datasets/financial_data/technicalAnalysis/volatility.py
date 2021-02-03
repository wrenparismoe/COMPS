import numpy as np
import pandas as pd

from technicalAnalysis.utils import IndicatorMixin, ema


class AverageTrueRange(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        cs = self._close.shift(1)
        tr = self._true_range(self._high, self._low, cs)
        atr = np.zeros(len(self._close))
        atr[self._n-1] = tr[0:self._n].mean()
        for i in range(self._n, len(atr)):
            atr[i] = (atr[i-1] * (self._n-1) + tr.iloc[i]) / float(self._n)
        self._atr = pd.Series(data=atr, index=tr.index)

    def average_true_range(self) -> pd.Series:
        atr = self._check_fillna(self._atr, value=0)
        return pd.Series(atr, name='atr')


class BollingerBands(IndicatorMixin):
    def __init__(self, close: pd.Series, n: int = 20, ndev: int = 2, fillna: bool = False):
        self._close = close
        self._n = n
        self._ndev = ndev
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._n
        self._mavg = self._close.rolling(self._n, min_periods=min_periods).mean()
        self._mstd = self._close.rolling(self._n, min_periods=min_periods).std(ddof=0)
        self._hband = self._mavg + self._ndev * self._mstd
        self._lband = self._mavg - self._ndev * self._mstd

    def bollinger_mavg(self) -> pd.Series:
        mavg = self._check_fillna(self._mavg, value=-1)
        return pd.Series(mavg, name='mavg')

    def bollinger_hband(self) -> pd.Series:
        hband = self._check_fillna(self._hband, value=-1)
        return pd.Series(hband, name='hband')

    def bollinger_lband(self) -> pd.Series:
        lband = self._check_fillna(self._lband, value=-1)
        return pd.Series(lband, name='lband')

    def bollinger_wband(self) -> pd.Series:
        wband = ((self._hband - self._lband) / self._mavg) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name='bbiwband')

    def bollinger_pband(self) -> pd.Series:
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name='bbipband')

    def bollinger_hband_indicator(self) -> pd.Series:
        hband = pd.Series(np.where(self._close > self._hband, 1.0, 0.0), index=self._close.index)
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, index=self._close.index, name='bbihband')

    def bollinger_lband_indicator(self) -> pd.Series:
        lband = pd.Series(np.where(self._close < self._lband, 1.0, 0.0), index=self._close.index)
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name='bbilband')


class KeltnerChannel(IndicatorMixin):
    def __init__(
            self, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, n_atr: int = 10,
            fillna: bool = False, ov: bool = True):
        self._high = high
        self._low = low
        self._close = close
        self._n = n
        self._n_atr = n_atr
        self._fillna = fillna
        self._ov = ov
        self._run()

    def _run(self):
        min_periods = 1 if self._fillna else self._n
        if self._ov:
            self._tp = ((self._high + self._low + self._close) / 3.0).rolling(self._n, min_periods=min_periods).mean()
            self._tp_high = (((4 * self._high) - (2 * self._low) + self._close) / 3.0).rolling(
                self._n, min_periods=0).mean()
            self._tp_low = (((-2 * self._high) + (4 * self._low) + self._close) / 3.0).rolling(
                self._n, min_periods=0).mean()
        else:
            self._tp = self._close.ewm(span=self._n, min_periods=min_periods, adjust=False).mean()
            atr = AverageTrueRange(
                close=self._close, high=self._high, low=self._low, n=self._n_atr, fillna=self._fillna
            ).average_true_range()
            self._tp_high = self._tp + (2*atr)
            self._tp_low = self._tp - (2*atr)

    def keltner_channel_mband(self) -> pd.Series:
        tp = self._check_fillna(self._tp, value=-1)
        return pd.Series(tp, name='mavg')

    def keltner_channel_hband(self) -> pd.Series:
        tp = self._check_fillna(self._tp_high, value=-1)
        return pd.Series(tp, name='kc_hband')

    def keltner_channel_lband(self) -> pd.Series:
        tp_low = self._check_fillna(self._tp_low, value=-1)
        return pd.Series(tp_low, name='kc_lband')

    def keltner_channel_wband(self) -> pd.Series:
        wband = ((self._tp_high - self._tp_low) / self._tp) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name='bbiwband')

    def keltner_channel_pband(self) -> pd.Series:
        pband = (self._close - self._tp_low) / (self._tp_high - self._tp_low)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name='bbipband')

    def keltner_channel_hband_indicator(self) -> pd.Series:
        hband = pd.Series(np.where(self._close > self._tp_high, 1.0, 0.0), index=self._close.index)
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, name='dcihband')

    def keltner_channel_lband_indicator(self) -> pd.Series:
        lband = pd.Series(np.where(self._close < self._tp_low, 1.0, 0.0), index=self._close.index)
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name='dcilband')


class DonchianChannel(IndicatorMixin):
    def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            n: int = 20,
            offset: int = 0,
            fillna: bool = False):
        self._offset = offset
        self._close = close
        self._high = high
        self._low = low
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        self._min_periods = 1 if self._fillna else self._n
        self._hband = self._high.rolling(self._n, min_periods=self._min_periods).max()
        self._lband = self._low.rolling(self._n, min_periods=self._min_periods).min()

    def donchian_channel_hband(self) -> pd.Series:
        hband = self._check_fillna(self._hband, value=-1)
        if self._offset != 0:
            hband = hband.shift(self._offset)
        return pd.Series(hband, name='dchband')

    def donchian_channel_lband(self) -> pd.Series:
        lband = self._check_fillna(self._lband, value=-1)
        if self._offset != 0:
            lband = lband.shift(self._offset)
        return pd.Series(lband, name='dclband')

    def donchian_channel_mband(self) -> pd.Series:
        mband = ((self._hband - self._lband) / 2.0) + self._lband
        mband = self._check_fillna(mband, value=-1)
        if self._offset != 0:
            mband = mband.shift(self._offset)
        return pd.Series(mband, name='dcmband')

    def donchian_channel_wband(self) -> pd.Series:
        mavg = self._close.rolling(self._n, min_periods=self._min_periods).mean()
        wband = ((self._hband - self._lband) / mavg) * 100
        wband = self._check_fillna(wband, value=0)
        if self._offset != 0:
            wband = wband.shift(self._offset)
        return pd.Series(wband, name='dcwband')

    def donchian_channel_pband(self) -> pd.Series:
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        if self._offset != 0:
            pband = pband.shift(self._offset)
        return pd.Series(pband, name='dcpband')


def average_true_range(high, low, close, n=14, fillna=False):
    indicator = AverageTrueRange(high=high, low=low, close=close, n=n, fillna=fillna)
    return indicator.average_true_range()


def bollinger_mavg(close, n=20, fillna=False):
    indicator = BollingerBands(close=close, n=n, fillna=fillna)
    return indicator.bollinger_mavg()


def bollinger_hband(close, n=20, ndev=2, fillna=False):
    indicator = BollingerBands(close=close, n=n, ndev=ndev, fillna=fillna)
    return indicator.bollinger_hband()


def bollinger_lband(close, n=20, ndev=2, fillna=False):
    indicator = BollingerBands(close=close, n=n, ndev=ndev, fillna=fillna)
    return indicator.bollinger_lband()


def bollinger_wband(close, n=20, ndev=2, fillna=False):
    indicator = BollingerBands(close=close, n=n, ndev=ndev, fillna=fillna)
    return indicator.bollinger_wband()


def bollinger_pband(close, n=20, ndev=2, fillna=False):
    indicator = BollingerBands(close=close, n=n, ndev=ndev, fillna=fillna)
    return indicator.bollinger_pband()


def bollinger_hband_indicator(close, n=20, ndev=2, fillna=False):
    indicator = BollingerBands(close=close, n=n, ndev=ndev, fillna=fillna)
    return indicator.bollinger_hband_indicator()


def bollinger_lband_indicator(close, n=20, ndev=2, fillna=False):
    indicator = BollingerBands(close=close, n=n, ndev=ndev, fillna=fillna)
    return indicator.bollinger_lband_indicator()


def keltner_channel_mband(high, low, close, n=20, n_atr=10, fillna=False, ov=True):
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, n_atr=n_atr, fillna=fillna, ov=ov)
    return indicator.keltner_channel_mband()


def keltner_channel_hband(high, low, close, n=20, n_atr=10, fillna=False, ov=True):
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, n_atr=n_atr, fillna=fillna, ov=ov)
    return indicator.keltner_channel_hband()


def keltner_channel_lband(high, low, close, n=20, n_atr=10, fillna=False, ov=True):
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, n_atr=n_atr, fillna=fillna, ov=ov)
    return indicator.keltner_channel_lband()


def keltner_channel_wband(high, low, close, n=20, n_atr=10, fillna=False, ov=True):
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, n_atr=n_atr, fillna=fillna, ov=ov)
    return indicator.keltner_channel_wband()


def keltner_channel_pband(high, low, close, n=20, n_atr=10, fillna=False, ov=True):
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, n_atr=n_atr, fillna=fillna, ov=ov)
    return indicator.keltner_channel_pband()


def keltner_channel_hband_indicator(high, low, close, n=20, n_atr=10, fillna=False, ov=True):
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, n_atr=n_atr, fillna=fillna, ov=ov)
    return indicator.keltner_channel_hband_indicator()


def keltner_channel_lband_indicator(high, low, close, n=20, n_atr=10, fillna=False, ov=True):
    indicator = KeltnerChannel(high=high, low=low, close=close, n=n, n_atr=n_atr, fillna=fillna, ov=ov)
    return indicator.keltner_channel_lband_indicator()


def donchian_channel_hband(high, low, close, n=20, offset=0, fillna=False):
    indicator = DonchianChannel(high=high, low=low, close=close, n=n, offset=offset, fillna=fillna)
    return indicator.donchian_channel_hband()


def donchian_channel_lband(high, low, close, n=20, offset=0, fillna=False):
    indicator = DonchianChannel(high=high, low=low, close=close, n=n, offset=offset, fillna=fillna)
    return indicator.donchian_channel_lband()


def donchian_channel_mband(high, low, close, n=10, offset=0, fillna=False):
    indicator = DonchianChannel(high=high, low=low, close=close, n=n, offset=offset, fillna=fillna)
    return indicator.donchian_channel_mband()


def donchian_channel_wband(high, low, close, n=10, offset=0, fillna=False):
    indicator = DonchianChannel(high=high, low=low, close=close, n=n, offset=offset, fillna=fillna)
    return indicator.donchian_channel_wband()


def donchian_channel_pband(high, low, close, n=10, offset=0, fillna=False):
    indicator = DonchianChannel(high=high, low=low, close=close, n=n, offset=offset, fillna=fillna)
    return indicator.donchian_channel_pband()