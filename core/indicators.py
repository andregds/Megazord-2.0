"""Indicators
Calcula e aplica indicadores técnicos (EMA, RSI, VWAP, ATR, VOL_MA) ao DataFrame.
Uso:
    df = Indicators.aplicar(df, ema_period=21)
"""

import numpy as np
import pandas as pd
from config import ATR_PERIOD, EMA_PERIOD

class Indicators:
    """Utilitários de indicadores técnicos aplicados sobre um DataFrame OHLCV."""

    @staticmethod
    def calcular_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Retorna o RSI da série de preços de fechamento."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calcular_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        """Retorna o ATR baseado em true range médio."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def aplicar(df: pd.DataFrame, ema_period: int = EMA_PERIOD) -> pd.DataFrame:
        """Adiciona colunas: EMA, RSI, VWAP, VOL_MA e ATR; retorna df sem NaNs iniciais."""
        df = df.copy()
        df["EMA"] = df["close"].ewm(span=ema_period, adjust=False).mean()
        df["RSI"] = Indicators.calcular_rsi(df["close"])
        df["VWAP"] = (df["close"] * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()
        df["VOL_MA"] = df["tick_volume"].rolling(20).mean()
        df["ATR"] = Indicators.calcular_atr(df)
        return df.dropna()
