import numpy as np
import pandas as pd

from technicalAnalysis.utils import IndicatorMixin


class RSIIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, n: int = 14, fillna: bool = False):
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        diff = self._close.diff(1)
        up = diff.where(diff > 0, 0.0)
        dn = -diff.where(diff < 0, 0.0)
        min_periods = 0 if self._fillna else self._n
        emaup = up.ewm(alpha=1/self._n, min_periods=min_periods, adjust=False).mean()
        emadn = dn.ewm(alpha=1/self._n, min_periods=min_periods, adjust=False).mean()
        rs = emaup / emadn
        self._rsi = pd.Series(np.where(emadn == 0, 100, 100-(100/(1+rs))), index=self._close.index)

    def rsi(self) -> pd.Series:

        rsi = self._check_fillna(self._rsi, value=50)
        return pd.Series(rsi, name='rsi')


class TSIIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, r: int = 25, s: int = 13, fillna: bool = False):
        self._close = close
        self._r = r
        self._s = s
        self._fillna = fillna
        self._run()

    def _run(self):
        m = self._close - self._close.shift(1)
        min_periods_r = 0 if self._fillna else self._r
        min_periods_s = 0 if self._fillna else self._s
        m1 = m.ewm(span=self._r, min_periods=min_periods_r, adjust=False).mean().ewm(
            span=self._s, min_periods=min_periods_s, adjust=False).mean()
        m2 = abs(m).ewm(span=self._r, min_periods=min_periods_r, adjust=False).mean().ewm(
            span=self._s, min_periods=min_periods_s, adjust=False).mean()
        self._tsi = m1 / m2
        self._tsi *= 100

    def tsi(self) -> pd.Series:
        tsi = self._check_fillna(self._tsi, value=0)
        return pd.Series(tsi, name='tsi')


class UltimateOscillator(IndicatorMixin):
    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 s: int = 7,
                 m: int = 14,
                 len: int = 28,
                 ws: float = 4.0,
                 wm: float = 2.0,
                 wl: float = 1.0,
                 fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._s = s
        self._m = m
        self._len = len
        self._ws = ws
        self._wm = wm
        self._wl = wl
        self._fillna = fillna
        self._run()

    def _run(self):
        cs = self._close.shift(1)
        tr = self._true_range(self._high, self._low, cs)
        bp = self._close - pd.DataFrame({'low': self._low, 'close': cs}).min(axis=1, skipna=False)
        min_periods_s = 0 if self._fillna else self._s
        min_periods_m = 0 if self._fillna else self._m
        min_periods_len = 0 if self._fillna else self._len
        avg_s = bp.rolling(
            self._s, min_periods=min_periods_s).sum() / tr.rolling(self._s, min_periods=min_periods_s).sum()
        avg_m = bp.rolling(
            self._m, min_periods=min_periods_m).sum() / tr.rolling(self._m, min_periods=min_periods_m).sum()
        avg_l = bp.rolling(
            self._len, min_periods=min_periods_len).sum() / tr.rolling(self._len, min_periods=min_periods_len).sum()
        self._uo = (100.0 * ((self._ws * avg_s) + (self._wm * avg_m) + (self._wl * avg_l))
                    / (self._ws + self._wm + self._wl))

    def uo(self) -> pd.Series:
        uo = self._check_fillna(self._uo, value=50)
        return pd.Series(uo, name='uo')


class StochasticOscillator(IndicatorMixin):
    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 n: int = 14,
                 d_n: int = 3,
                 fillna: bool = False):
        self._close = close
        self._high = high
        self._low = low
        self._n = n
        self._d_n = d_n
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._n
        smin = self._low.rolling(self._n, min_periods=min_periods).min()
        smax = self._high.rolling(self._n, min_periods=min_periods).max()
        self._stoch_k = 100 * (self._close - smin) / (smax - smin)

    def stoch(self) -> pd.Series:

        stoch_k = self._check_fillna(self._stoch_k, value=50)
        return pd.Series(stoch_k, name='stoch_k')

    def stoch_signal(self) -> pd.Series:
        min_periods = 0 if self._fillna else self._d_n
        stoch_d = self._stoch_k.rolling(self._d_n, min_periods=min_periods).mean()
        stoch_d = self._check_fillna(stoch_d, value=50)
        return pd.Series(stoch_d, name='stoch_k_signal')


class KAMAIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, n: int = 10, pow1: int = 2, pow2: int = 30, fillna: bool = False):
        self._close = close
        self._n = n
        self._pow1 = pow1
        self._pow2 = pow2
        self._fillna = fillna
        self._run()

    def _run(self):
        close_values = self._close.values
        vol = pd.Series(abs(self._close - np.roll(self._close, 1)))

        min_periods = 0 if self._fillna else self._n
        ER_num = abs(close_values - np.roll(close_values, self._n))
        ER_den = vol.rolling(self._n, min_periods=min_periods).sum()
        ER = ER_num / ER_den

        sc = ((ER*(2.0/(self._pow1+1)-2.0/(self._pow2+1.0))+2/(self._pow2+1.0)) ** 2.0).values

        self._kama = np.zeros(sc.size)
        n = len(self._kama)
        first_value = True

        for i in range(n):
            if np.isnan(sc[i]):
                self._kama[i] = np.nan
            else:
                if first_value:
                    self._kama[i] = close_values[i]
                    first_value = False
                else:
                    self._kama[i] = self._kama[i-1] + sc[i] * (close_values[i] - self._kama[i-1])

    def kama(self) -> pd.Series:
        kama = pd.Series(self._kama, index=self._close.index)
        kama = self._check_fillna(kama, value=self._close)
        return pd.Series(kama, name='kama')


class ROCIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, n: int = 12, fillna: bool = False):
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        self._roc = ((self._close - self._close.shift(self._n)) / self._close.shift(self._n)) * 100

    def roc(self) -> pd.Series:
        roc = self._check_fillna(self._roc)
        return pd.Series(roc, name='roc')


class AwesomeOscillatorIndicator(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, s: int = 5, len: int = 34, fillna: bool = False):
        self._high = high
        self._low = low
        self._s = s
        self._len = len
        self._fillna = fillna
        self._run()

    def _run(self):
        mp = 0.5 * (self._high + self._low)
        min_periods_s = 0 if self._fillna else self._s
        min_periods_len = 0 if self._fillna else self._len
        self._ao = mp.rolling(
            self._s, min_periods=min_periods_s).mean() - mp.rolling(self._len, min_periods=min_periods_len).mean()

    def ao(self) -> pd.Series:
        ao = self._check_fillna(self._ao, value=0)
        return pd.Series(ao, name='ao')


class WilliamsRIndicator(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, lbp: int = 14, fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._lbp = lbp
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._lbp
        hh = self._high.rolling(self._lbp, min_periods=min_periods).max()  # highest high over lookback period lbp
        ll = self._low.rolling(self._lbp, min_periods=min_periods).min()  # lowest low over lookback period lbp
        self._wr = -100 * (hh - self._close) / (hh - ll)

    def wr(self) -> pd.Series:
        wr = self._check_fillna(self._wr, value=-50)
        return pd.Series(wr, name='wr')


def rsi(close, n=14, fillna=False):
    return RSIIndicator(close=close, n=n, fillna=fillna).rsi()


def tsi(close, r=25, s=13, fillna=False):
    return TSIIndicator(close=close, r=r, s=s, fillna=fillna).tsi()


def uo(high, low, close, s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0, fillna=False):
    return UltimateOscillator(
        high=high, low=low, close=close, s=s, m=m, len=len, ws=ws, wm=wm, wl=wl, fillna=fillna).uo()


def stoch(high, low, close, n=14, d_n=3, fillna=False):
    return StochasticOscillator(high=high, low=low, close=close, n=n, d_n=d_n, fillna=fillna).stoch()


def stoch_signal(high, low, close, n=14, d_n=3, fillna=False):
    return StochasticOscillator(high=high, low=low, close=close, n=n, d_n=d_n, fillna=fillna).stoch_signal()


def wr(high, low, close, lbp=14, fillna=False):
    return WilliamsRIndicator(high=high, low=low, close=close, lbp=lbp, fillna=fillna).wr()


def ao(high, low, s=5, len=34, fillna=False):
    return AwesomeOscillatorIndicator(high=high, low=low, s=s, len=len, fillna=fillna).ao()


def kama(close, n=10, pow1=2, pow2=30, fillna=False):
    return KAMAIndicator(close=close, n=n, pow1=pow1, pow2=pow2, fillna=fillna).kama()


def roc(close, n=12, fillna=False):
    return ROCIndicator(close=close, n=n, fillna=fillna).roc()