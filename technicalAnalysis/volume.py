import numpy as np
import pandas as pd

from technicalAnalysis.utils import IndicatorMixin, ema


class AccDistIndexIndicator(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        clv = ((self._close - self._low) - (self._high - self._close)) / (self._high - self._low)
        clv = clv.fillna(0.0)  # float division by zero
        ad = clv * self._volume
        self._ad = ad.cumsum()

    def acc_dist_index(self) -> pd.Series:
        ad = self._check_fillna(self._ad, value=0)
        return pd.Series(ad, name='adi')


class OnBalanceVolumeIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        obv = np.where(self._close < self._close.shift(1), -self._volume, self._volume)
        self._obv = pd.Series(obv, index=self._close.index).cumsum()

    def on_balance_volume(self) -> pd.Series:
        obv = self._check_fillna(self._obv, value=0)
        return pd.Series(obv, name='obv')


class ChaikinMoneyFlowIndicator(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, n: int = 20, fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        mfv = ((self._close - self._low) - (self._high - self._close)) / (self._high - self._low)
        mfv = mfv.fillna(0.0)  # float division by zero
        mfv *= self._volume
        min_periods = 0 if self._fillna else self._n
        self._cmf = (
                mfv.rolling(self._n, min_periods=min_periods).sum() /
                self._volume.rolling(self._n, min_periods=min_periods).sum())

    def chaikin_money_flow(self) -> pd.Series:
        cmf = self._check_fillna(self._cmf, value=0)
        return pd.Series(cmf, name='cmf')


class ForceIndexIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, volume: pd.Series, n: int = 13, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        fi = (self._close - self._close.shift(1)) * self._volume
        self._fi = ema(fi, self._n, fillna=self._fillna)

    def force_index(self) -> pd.Series:
        fi = self._check_fillna(self._fi, value=0)
        return pd.Series(fi, name=f'fi_{self._n}')


class EaseOfMovementIndicator(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, volume: pd.Series, n: int = 14, fillna: bool = False):
        self._high = high
        self._low = low
        self._volume = volume
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        self._emv = (self._high.diff(1) + self._low.diff(1)) * (self._high - self._low) / (2 * self._volume)
        self._emv *= 100000000

    def ease_of_movement(self) -> pd.Series:
        emv = self._check_fillna(self._emv, value=0)
        return pd.Series(emv, name=f'eom_{self._n}')

    def sma_ease_of_movement(self) -> pd.Series:
        min_periods = 0 if self._fillna else self._n
        emv = self._emv.rolling(self._n, min_periods=min_periods).mean()
        emv = self._check_fillna(emv, value=0)
        return pd.Series(emv, name=f'sma_eom_{self._n}')


class VolumePriceTrendIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        vpt = (self._volume * ((self._close - self._close.shift(1, fill_value=self._close.mean()))
                               / self._close.shift(1, fill_value=self._close.mean())))
        self._vpt = vpt.shift(1, fill_value=vpt.mean()) + vpt

    def volume_price_trend(self) -> pd.Series:
        vpt = self._check_fillna(self._vpt, value=0)
        return pd.Series(vpt, name='vpt')


class NegativeVolumeIndexIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        price_change = self._close.pct_change()
        vol_decrease = (self._volume.shift(1) > self._volume)
        self._nvi = pd.Series(data=np.nan, index=self._close.index, dtype='float64', name='nvi')
        self._nvi.iloc[0] = 1000
        for i in range(1, len(self._nvi)):
            if vol_decrease.iloc[i]:
                self._nvi.iloc[i] = self._nvi.iloc[i - 1] * (1.0 + price_change.iloc[i])
            else:
                self._nvi.iloc[i] = self._nvi.iloc[i - 1]

    def negative_volume_index(self) -> pd.Series:
        # IDEA: There shouldn't be any na; might be better to throw exception
        nvi = self._check_fillna(self._nvi, value=1000)
        return pd.Series(nvi, name='nvi')


class MFIIndicator(IndicatorMixin):
    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 volume: pd.Series,
                 n: int = 14,
                 fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        # 1 typical price
        tp = (self._high + self._low + self._close) / 3.0

        # 2 up or down column
        up_down = np.where(tp > tp.shift(1), 1, np.where(tp < tp.shift(1), -1, 0))

        # 3 money flow
        mf = tp * self._volume * up_down

        # 4 positive and negative money flow with n periods
        min_periods = 0 if self._fillna else self._n
        n_positive_mf = mf.rolling(
            self._n, min_periods=min_periods).apply(lambda x: np.sum(np.where(x >= 0.0, x, 0.0)), raw=True)
        n_negative_mf = abs(
            mf.rolling(self._n, min_periods=min_periods).apply(lambda x: np.sum(np.where(x < 0.0, x, 0.0)), raw=True))

        # n_positive_mf = np.where(mf.rolling(self._n).sum() >= 0.0, mf, 0.0)
        # n_negative_mf = abs(np.where(mf.rolling(self._n).sum() < 0.0, mf, 0.0))

        # 5 money flow index
        mr = n_positive_mf / n_negative_mf
        self._mr = (100 - (100 / (1 + mr)))

    def money_flow_index(self) -> pd.Series:
        mr = self._check_fillna(self._mr, value=50)
        return pd.Series(mr, name=f'mfi_{self._n}')


class VolumeWeightedAveragePrice(IndicatorMixin):
    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 volume: pd.Series,
                 n: int = 14,
                 fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        # 1 typical price
        tp = (self._high + self._low + self._close) / 3.0

        # 2 typical price * volume
        pv = (tp * self._volume)

        # 3 total price * volume
        min_periods = 0 if self._fillna else self._n
        total_pv = pv.rolling(self._n, min_periods=min_periods).sum()

        # 4 total volume
        total_volume = self._volume.rolling(self._n, min_periods=min_periods).sum()

        self.vwap = total_pv / total_volume

    def volume_weighted_average_price(self) -> pd.Series:
        vwap = self._check_fillna(self.vwap)
        return pd.Series(vwap, name=f'vwap_{self._n}')


def acc_dist_index(high, low, close, volume, fillna=False):
    return AccDistIndexIndicator(high=high, low=low, close=close, volume=volume, fillna=fillna).acc_dist_index()


def on_balance_volume(close, volume, fillna=False):
    return OnBalanceVolumeIndicator(close=close, volume=volume, fillna=fillna).on_balance_volume()


def chaikin_money_flow(high, low, close, volume, n=20, fillna=False):
    return ChaikinMoneyFlowIndicator(
        high=high, low=low, close=close, volume=volume, n=n, fillna=fillna).chaikin_money_flow()


def force_index(close, volume, n=13, fillna=False):
    return ForceIndexIndicator(close=close, volume=volume, n=n, fillna=fillna).force_index()


def ease_of_movement(high, low, volume, n=14, fillna=False):
    return EaseOfMovementIndicator(
        high=high, low=low, volume=volume, n=n, fillna=fillna).ease_of_movement()


def sma_ease_of_movement(high, low, volume, n=14, fillna=False):
    return EaseOfMovementIndicator(
        high=high, low=low, volume=volume, n=n, fillna=fillna).sma_ease_of_movement()


def volume_price_trend(close, volume, fillna=False):
    return VolumePriceTrendIndicator(close=close, volume=volume, fillna=fillna).volume_price_trend()


def negative_volume_index(close, volume, fillna=False):
    return NegativeVolumeIndexIndicator(close=close, volume=volume, fillna=fillna).negative_volume_index()


def money_flow_index(high, low, close, volume, n=14, fillna=False):
    indicator = MFIIndicator(high=high, low=low, close=close, volume=volume, n=n, fillna=fillna)
    return indicator.money_flow_index()


def volume_weighted_average_price(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14, fillna: bool = False):
    indicator = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, n=n, fillna=fillna)
    return indicator.volume_weighted_average_price()