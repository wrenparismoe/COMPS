import numpy as np
import pandas as pd

from technicalAnalysis.utils import IndicatorMixin, ema, get_min_max, sma


class AroonIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, n: int = 25, fillna: bool = False):
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._n
        rolling_close = self._close.rolling(self._n, min_periods=min_periods)
        self._aroon_up = rolling_close.apply(
            lambda x: float(np.argmax(x) + 1) / self._n * 100, raw=True)
        self._aroon_down = rolling_close.apply(
            lambda x: float(np.argmin(x) + 1) / self._n * 100, raw=True)

    def aroon_up(self) -> pd.Series:
        aroon_up = self._check_fillna(self._aroon_up, value=0)
        return pd.Series(aroon_up, name=f'aroon_up_{self._n}')

    def aroon_down(self) -> pd.Series:
        aroon_down = self._check_fillna(self._aroon_down, value=0)
        return pd.Series(aroon_down, name=f'aroon_down_{self._n}')

    def aroon_indicator(self) -> pd.Series:
        aroon_diff = self._aroon_up - self._aroon_down
        aroon_diff = self._check_fillna(aroon_diff, value=0)
        return pd.Series(aroon_diff, name=f'aroon_ind_{self._n}')


class MACD(IndicatorMixin):
    def __init__(self,
                 close: pd.Series,
                 n_slow: int = 26,
                 n_fast: int = 12,
                 n_sign: int = 9,
                 fillna: bool = False):
        self._close = close
        self._n_slow = n_slow
        self._n_fast = n_fast
        self._n_sign = n_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        self._emafast = ema(self._close, self._n_fast, self._fillna)
        self._emaslow = ema(self._close, self._n_slow, self._fillna)
        self._macd = self._emafast - self._emaslow
        self._macd_signal = ema(self._macd, self._n_sign, self._fillna)
        self._macd_diff = self._macd - self._macd_signal

    def macd(self) -> pd.Series:
        macd = self._check_fillna(self._macd, value=0)
        return pd.Series(macd, name=f'MACD_{self._n_fast}_{self._n_slow}')

    def macd_signal(self) -> pd.Series:
        macd_signal = self._check_fillna(self._macd_signal, value=0)
        return pd.Series(macd_signal, name=f'MACD_sign_{self._n_fast}_{self._n_slow}')

    def macd_diff(self) -> pd.Series:
        macd_diff = self._check_fillna(self._macd_diff, value=0)
        return pd.Series(macd_diff, name=f'MACD_diff_{self._n_fast}_{self._n_slow}')


class EMAIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, n: int = 14, fillna: bool = False):
        self._close = close
        self._n = n
        self._fillna = fillna

    def ema_indicator(self) -> pd.Series:
        ema_ = ema(self._close, self._n, self._fillna)
        return pd.Series(ema_, name=f'ema_{self._n}')


class SMAIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, n: int, fillna: bool = False):
        self._close = close
        self._n = n
        self._fillna = fillna

    def sma_indicator(self) -> pd.Series:
        sma_ = sma(self._close, self._n, self._fillna)
        return pd.Series(sma_, name=f'sma_{self._n}')


class TRIXIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, n: int = 15, fillna: bool = False):
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        ema1 = ema(self._close, self._n, self._fillna)
        ema2 = ema(ema1, self._n, self._fillna)
        ema3 = ema(ema2, self._n, self._fillna)
        self._trix = (ema3 - ema3.shift(1, fill_value=ema3.mean())) / ema3.shift(1, fill_value=ema3.mean())
        self._trix *= 100

    def trix(self) -> pd.Series:
        trix = self._check_fillna(self._trix, value=0)
        return pd.Series(trix, name=f'trix_{self._n}')


class MassIndex(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, n: int = 9, n2: int = 25, fillna: bool = False):
        self._high = high
        self._low = low
        self._n = n
        self._n2 = n2
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._n2
        amplitude = self._high - self._low
        ema1 = ema(amplitude, self._n, self._fillna)
        ema2 = ema(ema1, self._n, self._fillna)
        mass = ema1 / ema2
        self._mass = mass.rolling(self._n2, min_periods=min_periods).sum()

    def mass_index(self) -> pd.Series:
        mass = self._check_fillna(self._mass, value=0)
        return pd.Series(mass, name=f'mass_index_{self._n}_{self._n2}')


class IchimokuIndicator(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, n1: int = 9, n2: int = 26, n3: int = 52,
                 visual: bool = False, fillna: bool = False):
        self._high = high
        self._low = low
        self._n1 = n1
        self._n2 = n2
        self._n3 = n3
        self._visual = visual
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods_n1 = 0 if self._fillna else self._n1
        min_periods_n2 = 0 if self._fillna else self._n2
        self._conv = 0.5 * (
                self._high.rolling(self._n1, min_periods=min_periods_n1).max() +
                self._low.rolling(self._n1, min_periods=min_periods_n1).min())
        self._base = 0.5 * (
                self._high.rolling(self._n2, min_periods=min_periods_n2).max() +
                self._low.rolling(self._n2, min_periods=min_periods_n2).min())

    def ichimoku_conversion_line(self) -> pd.Series:
        conversion = self._check_fillna(self._conv, value=-1)
        return pd.Series(conversion, name=f'ichimoku_conv_{self._n1}_{self._n2}')

    def ichimoku_base_line(self) -> pd.Series:
        base = self._check_fillna(self._base, value=-1)
        return pd.Series(base, name=f'ichimoku_base_{self._n1}_{self._n2}')

    def ichimoku_a(self) -> pd.Series:
        spana = 0.5 * (self._conv + self._base)
        spana = spana.shift(self._n2, fill_value=spana.mean()) if self._visual else spana
        spana = self._check_fillna(spana, value=-1)
        return pd.Series(spana, name=f'ichimoku_a_{self._n1}_{self._n2}')

    def ichimoku_b(self) -> pd.Series:
        spanb = 0.5 * (self._high.rolling(self._n3, min_periods=0).max()
                       + self._low.rolling(self._n3, min_periods=0).min())
        spanb = spanb.shift(self._n2, fill_value=spanb.mean()) if self._visual else spanb
        spanb = self._check_fillna(spanb, value=-1)
        return pd.Series(spanb, name=f'ichimoku_b_{self._n1}_{self._n2}')


class KSTIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30,
                 n1: int = 10, n2: int = 10, n3: int = 10, n4: int = 15, nsig: int = 9,
                 fillna: bool = False):
        self._close = close
        self._r1 = r1
        self._r2 = r2
        self._r3 = r3
        self._r4 = r4
        self._n1 = n1
        self._n2 = n2
        self._n3 = n3
        self._n4 = n4
        self._nsig = nsig
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods_n1 = 0 if self._fillna else self._n1
        min_periods_n2 = 0 if self._fillna else self._n2
        min_periods_n3 = 0 if self._fillna else self._n3
        min_periods_n4 = 0 if self._fillna else self._n4
        rocma1 = (
            (self._close - self._close.shift(self._r1, fill_value=self._close.mean()))
            / self._close.shift(self._r1, fill_value=self._close.mean())).rolling(
                self._n1, min_periods=min_periods_n1).mean()
        rocma2 = (
            (self._close - self._close.shift(self._r2, fill_value=self._close.mean()))
            / self._close.shift(self._r2, fill_value=self._close.mean())).rolling(
                self._n2, min_periods=min_periods_n2).mean()
        rocma3 = (
            (self._close - self._close.shift(self._r3, fill_value=self._close.mean()))
            / self._close.shift(self._r3, fill_value=self._close.mean())).rolling(
                self._n3, min_periods=min_periods_n3).mean()
        rocma4 = (
            (self._close - self._close.shift(self._r4, fill_value=self._close.mean()))
            / self._close.shift(self._r4, fill_value=self._close.mean())).rolling(
                self._n4, min_periods=min_periods_n4).mean()
        self._kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
        self._kst_sig = self._kst.rolling(self._nsig, min_periods=0).mean()

    def kst(self) -> pd.Series:
        kst = self._check_fillna(self._kst, value=0)
        return pd.Series(kst, name='kst')

    def kst_sig(self) -> pd.Series:
        kst_sig = self._check_fillna(self._kst_sig, value=0)
        return pd.Series(kst_sig, name='kst_sig')

    def kst_diff(self) -> pd.Series:
        kst_diff = self._kst - self._kst_sig
        kst_diff = self._check_fillna(kst_diff, value=0)
        return pd.Series(kst_diff, name='kst_diff')


class DPOIndicator(IndicatorMixin):
    def __init__(self, close: pd.Series, n: int = 20, fillna: bool = False):
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._n
        self._dpo = (self._close.shift(int((0.5 * self._n) + 1), fill_value=self._close.mean())
                     - self._close.rolling(self._n, min_periods=min_periods).mean())

    def dpo(self) -> pd.Series:
        dpo = self._check_fillna(self._dpo, value=0)
        return pd.Series(dpo, name='dpo_'+str(self._n))


class CCIIndicator(IndicatorMixin):
    def __init__(self,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series,
                 n: int = 20,
                 c: float = 0.015,
                 fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._n = n
        self._c = c
        self._fillna = fillna
        self._run()

    def _run(self):

        def _mad(x):
            return np.mean(np.abs(x-np.mean(x)))

        min_periods = 0 if self._fillna else self._n
        pp = (self._high + self._low + self._close) / 3.0
        self._cci = ((pp - pp.rolling(self._n, min_periods=min_periods).mean())
                     / (self._c * pp.rolling(self._n, min_periods=min_periods).apply(_mad, True)))

    def cci(self) -> pd.Series:

        cci = self._check_fillna(self._cci, value=0)
        return pd.Series(cci, name='cci')


class ADXIndicator(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        if self._n == 0:
            raise ValueError("window may not be 0")

        cs = self._close.shift(1)
        pdm = get_min_max(self._high, cs, 'max')
        pdn = get_min_max(self._low, cs, 'min')
        tr = pdm - pdn

        self._trs_initial = np.zeros(self._n-1)
        self._trs = np.zeros(len(self._close) - (self._n - 1))
        self._trs[0] = tr.dropna()[0:self._n].sum()
        tr = tr.reset_index(drop=True)

        for i in range(1, len(self._trs)-1):
            self._trs[i] = self._trs[i-1] - (self._trs[i-1]/float(self._n)) + tr[self._n+i]

        up = self._high - self._high.shift(1)
        dn = self._low.shift(1) - self._low
        pos = abs(((up > dn) & (up > 0)) * up)
        neg = abs(((dn > up) & (dn > 0)) * dn)

        self._dip = np.zeros(len(self._close) - (self._n - 1))
        self._dip[0] = pos.dropna()[0:self._n].sum()

        pos = pos.reset_index(drop=True)

        for i in range(1, len(self._dip)-1):
            self._dip[i] = self._dip[i-1] - (self._dip[i-1]/float(self._n)) + pos[self._n+i]

        self._din = np.zeros(len(self._close) - (self._n - 1))
        self._din[0] = neg.dropna()[0:self._n].sum()

        neg = neg.reset_index(drop=True)

        for i in range(1, len(self._din)-1):
            self._din[i] = self._din[i-1] - (self._din[i-1]/float(self._n)) + neg[self._n+i]

    def adx(self) -> pd.Series:
        dip = np.zeros(len(self._trs))
        for i in range(len(self._trs)):
            dip[i] = 100 * (self._dip[i]/self._trs[i])

        din = np.zeros(len(self._trs))
        for i in range(len(self._trs)):
            din[i] = 100 * (self._din[i]/self._trs[i])

        dx = 100 * np.abs((dip - din) / (dip + din))

        adx = np.zeros(len(self._trs))
        adx[self._n] = dx[0:self._n].mean()

        for i in range(self._n+1, len(adx)):
            adx[i] = ((adx[i-1] * (self._n - 1)) + dx[i-1]) / float(self._n)

        adx = np.concatenate((self._trs_initial, adx), axis=0)
        self._adx = pd.Series(data=adx, index=self._close.index)

        adx = self._check_fillna(self._adx, value=20)
        return pd.Series(adx, name='adx')

    def adx_pos(self) -> pd.Series:
        dip = np.zeros(len(self._close))
        for i in range(1, len(self._trs)-1):
            dip[i+self._n] = 100 * (self._dip[i]/self._trs[i])

        adx_pos = self._check_fillna(pd.Series(dip, index=self._close.index), value=20)
        return pd.Series(adx_pos, name='adx_pos')

    def adx_neg(self) -> pd.Series:
        din = np.zeros(len(self._close))
        for i in range(1, len(self._trs)-1):
            din[i+self._n] = 100 * (self._din[i]/self._trs[i])

        adx_neg = self._check_fillna(pd.Series(din, index=self._close.index), value=20)
        return pd.Series(adx_neg, name='adx_neg')


class VortexIndicator(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._n = n
        self._fillna = fillna
        self._run()

    def _run(self):
        cs = self._close.shift(1, fill_value=self._close.mean())
        tr = self._true_range(self._high, self._low, cs)
        min_periods = 0 if self._fillna else self._n
        trn = tr.rolling(self._n, min_periods=min_periods).sum()
        vmp = np.abs(self._high - self._low.shift(1))
        vmm = np.abs(self._low - self._high.shift(1))
        self._vip = vmp.rolling(self._n, min_periods=min_periods).sum() / trn
        self._vin = vmm.rolling(self._n, min_periods=min_periods).sum() / trn

    def vortex_indicator_pos(self):
        vip = self._check_fillna(self._vip, value=1)
        return pd.Series(vip, name='vip')

    def vortex_indicator_neg(self):
        vin = self._check_fillna(self._vin, value=1)
        return pd.Series(vin, name='vin')

    def vortex_indicator_diff(self):
        vid = self._vip - self._vin
        vid = self._check_fillna(vid, value=0)
        return pd.Series(vid, name='vid')


class PSARIndicator(IndicatorMixin):
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series,
                 step: float = 0.02, max_step: float = 0.20,
                 fillna: bool = False):
        self._high = high
        self._low = low
        self._close = close
        self._step = step
        self._max_step = max_step
        self._fillna = fillna
        self._run()

    def _run(self):
        up_trend = True
        af = self._step
        up_trend_high = self._high.iloc[0]
        down_trend_low = self._low.iloc[0]

        self._psar = self._close.copy()
        self._psar_up = pd.Series(index=self._psar.index)
        self._psar_down = pd.Series(index=self._psar.index)

        for i in range(2, len(self._close)):
            reversal = False

            max_high = self._high.iloc[i]
            min_low = self._low.iloc[i]

            if up_trend:
                self._psar.iloc[i] = self._psar.iloc[i-1] + (
                    af * (up_trend_high - self._psar.iloc[i-1]))

                if min_low < self._psar.iloc[i]:
                    reversal = True
                    self._psar.iloc[i] = up_trend_high
                    down_trend_low = min_low
                    af = self._step
                else:
                    if max_high > up_trend_high:
                        up_trend_high = max_high
                        af = min(af + self._step, self._max_step)

                    l1 = self._low.iloc[i-1]
                    l2 = self._low.iloc[i-2]
                    if l2 < self._psar.iloc[i]:
                        self._psar.iloc[i] = l2
                    elif l1 < self._psar.iloc[i]:
                        self._psar.iloc[i] = l1
            else:
                self._psar.iloc[i] = self._psar.iloc[i-1] - (
                    af * (self._psar.iloc[i-1] - down_trend_low))

                if max_high > self._psar.iloc[i]:
                    reversal = True
                    self._psar.iloc[i] = down_trend_low
                    up_trend_high = max_high
                    af = self._step
                else:
                    if min_low < down_trend_low:
                        down_trend_low = min_low
                        af = min(af + self._step, self._max_step)

                    h1 = self._high.iloc[i-1]
                    h2 = self._high.iloc[i-2]
                    if h2 > self._psar.iloc[i]:
                        self._psar[i] = h2
                    elif h1 > self._psar.iloc[i]:
                        self._psar.iloc[i] = h1

            up_trend = up_trend != reversal  # XOR

            if up_trend:
                self._psar_up.iloc[i] = self._psar.iloc[i]
            else:
                self._psar_down.iloc[i] = self._psar.iloc[i]

    def psar(self) -> pd.Series:
        psar = self._check_fillna(self._psar, value=-1)
        return pd.Series(psar, name='psar')

    def psar_up(self) -> pd.Series:
        psar_up = self._check_fillna(self._psar_up, value=-1)
        return pd.Series(psar_up, name='psarup')

    def psar_down(self) -> pd.Series:
        psar_down = self._check_fillna(self._psar_down, value=-1)
        return pd.Series(psar_down, name='psardown')

    def psar_up_indicator(self) -> pd.Series:
        indicator = self._psar_up.where(self._psar_up.notnull()
                                        & self._psar_up.shift(1).isnull(), 0)
        indicator = indicator.where(indicator == 0, 1)
        return pd.Series(indicator, index=self._close.index, name='psariup')

    def psar_down_indicator(self) -> pd.Series:
        indicator = self._psar_up.where(self._psar_down.notnull()
                                        & self._psar_down.shift(1).isnull(), 0)
        indicator = indicator.where(indicator == 0, 1)
        return pd.Series(indicator, index=self._close.index, name='psaridown')


def ema_indicator(close, n=12, fillna=False):
    return EMAIndicator(close=close, n=n, fillna=fillna).ema_indicator()


def sma_indicator(close, n=12, fillna=False):
    return SMAIndicator(close=close, n=n, fillna=fillna).sma_indicator()


def macd(close, n_slow=26, n_fast=12, fillna=False):
    return MACD(close=close, n_slow=n_slow, n_fast=n_fast, n_sign=9, fillna=fillna).macd()


def macd_signal(close, n_slow=26, n_fast=12, n_sign=9, fillna=False):
    return MACD(close=close, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna).macd_signal()


def macd_diff(close, n_slow=26, n_fast=12, n_sign=9, fillna=False):
    return MACD(close=close, n_slow=n_slow, n_fast=n_fast, n_sign=n_sign, fillna=fillna).macd_diff()


def adx(high, low, close, n=14, fillna=False):
    return ADXIndicator(high=high, low=low, close=close, n=n, fillna=fillna).adx()


def adx_pos(high, low, close, n=14, fillna=False):
    return ADXIndicator(high=high, low=low, close=close, n=n, fillna=fillna).adx_pos()


def adx_neg(high, low, close, n=14, fillna=False):
    return ADXIndicator(high=high, low=low, close=close, n=n, fillna=fillna).adx_neg()


def vortex_indicator_pos(high, low, close, n=14, fillna=False):
    return VortexIndicator(high=high, low=low, close=close, n=n, fillna=fillna).vortex_indicator_pos()


def vortex_indicator_neg(high, low, close, n=14, fillna=False):
    return VortexIndicator(high=high, low=low, close=close, n=n, fillna=fillna).vortex_indicator_neg()


def trix(close, n=15, fillna=False):
    return TRIXIndicator(close=close, n=n, fillna=fillna).trix()


def mass_index(high, low, n=9, n2=25, fillna=False):
    return MassIndex(high=high, low=low, n=n, n2=n2, fillna=fillna).mass_index()


def cci(high, low, close, n=20, c=0.015, fillna=False):
    return CCIIndicator(high=high, low=low, close=close, n=n, c=c, fillna=fillna).cci()


def dpo(close, n=20, fillna=False):
    return DPOIndicator(close=close, n=n, fillna=fillna).dpo()


def kst(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, fillna=False):
    return KSTIndicator(
        close=close, r1=r1, r2=r2, r3=r3, r4=r4, n1=n1, n2=n2, n3=n3, n4=n4, nsig=9, fillna=fillna).kst()


def kst_sig(close, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=False):
    return KSTIndicator(
        close=close, r1=r1, r2=r2, r3=r3, r4=r4, n1=n1, n2=n2, n3=n3, n4=n4, nsig=nsig, fillna=fillna).kst_sig()


def ichimoku_conversion_line(high, low, n1=9, n2=26, visual=False, fillna=False) -> pd.Series:
    return IchimokuIndicator(
        high=high, low=low, n1=n1, n2=n2, n3=52, visual=visual, fillna=fillna).ichimoku_conversion_line()


def ichimoku_base_line(high, low, n1=9, n2=26, visual=False, fillna=False) -> pd.Series:
    return IchimokuIndicator(
        high=high, low=low, n1=n1, n2=n2, n3=52, visual=visual, fillna=fillna).ichimoku_base_line()


def ichimoku_a(high, low, n1=9, n2=26, visual=False, fillna=False):
    return IchimokuIndicator(high=high, low=low, n1=n1, n2=n2, n3=52, visual=visual, fillna=fillna).ichimoku_a()


def ichimoku_b(high, low, n2=26, n3=52, visual=False, fillna=False):
    return IchimokuIndicator(high=high, low=low, n1=9, n2=n2, n3=n3, visual=visual, fillna=fillna).ichimoku_b()


def aroon_up(close, n=25, fillna=False):
    return AroonIndicator(close=close, n=n, fillna=fillna).aroon_up()


def aroon_down(close, n=25, fillna=False):
    return AroonIndicator(close=close, n=n, fillna=fillna).aroon_down()


def psar_up(high, low, close, step=0.02, max_step=0.20, fillna=False):
    indicator = PSARIndicator(high=high, low=low, close=close, step=step,
                              max_step=max_step, fillna=fillna)
    return indicator.psar_up()


def psar_down(high, low, close, step=0.02, max_step=0.20, fillna=False):
    indicator = PSARIndicator(high=high, low=low, close=close, step=step,
                              max_step=max_step, fillna=fillna)
    return indicator.psar_down()


def psar_up_indicator(high, low, close, step=0.02, max_step=0.20, fillna=False):
    indicator = PSARIndicator(high=high, low=low, close=close, step=step,
                              max_step=max_step, fillna=fillna)
    return indicator.psar_up_indicator()


def psar_down_indicator(high, low, close, step=0.02, max_step=0.20, fillna=False):
    indicator = PSARIndicator(high=high, low=low, close=close, step=step,
                              max_step=max_step, fillna=fillna)
    return indicator.psar_down_indicator()