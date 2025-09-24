"""Analyzer
Consolida indicadores + padrão para gerar probabilidade e níveis de trade.
Também avalia, no futuro, se o trade bateu alvo/stop.
Uso:
    prob, sinal = Analyzer.calcular_probabilidade(df, padrao)
    entry, stop, target, atr = Analyzer.montar_trade_levels(df, sinal)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from config import (
    RSI_SOBRECOMPRA, RSI_SOBREVENDIDO, ATR_MULT_STOP, RR_MULT,
    EVAL_BARS, TIMEFRAMES
)

class Analyzer:
    """Gera probabilidade/sinal, níveis de trade (entry/stop/target) e avalia resultado."""

    @staticmethod
    def calcular_probabilidade(df: pd.DataFrame, padrao: str,
                               rsi_buy: float = RSI_SOBREVENDIDO,
                               rsi_sell: float = RSI_SOBRECOMPRA) -> tuple[int, str]:
        """Retorna (score %, sinal 'COMPRA'/'VENDA') usando filtros (RSI/EMA/VWAP/Volume)."""
        rsi   = df["RSI"].iloc[-1]
        ema   = df["EMA"].iloc[-1]
        vwap  = df["VWAP"].iloc[-1]
        preco = df["close"].iloc[-1]
        vol   = df["tick_volume"].iloc[-1]
        vol_ma= df["VOL_MA"].iloc[-1]

        score = 0
        if padrao == "Double Bottom":
            if rsi < rsi_buy: score += 30
            if preco > ema:   score += 30
            if vol > vol_ma:  score += 20
            if preco > vwap:  score += 20
            sinal = "COMPRA"
        else:
            if rsi > rsi_sell: score += 30
            if preco < ema:    score += 30
            if vol > vol_ma:   score += 20
            if preco < vwap:   score += 20
            sinal = "VENDA"

        return score, sinal

    @staticmethod
    def montar_trade_levels(df: pd.DataFrame, sinal: str,
                            atr_mult: float = ATR_MULT_STOP,
                            rr: float = RR_MULT) -> tuple[float, float, float, float]:
        """Calcula entry/stop/target com base no ATR; retorna (entry, stop, target, atr)."""
        entry = float(df["close"].iloc[-1])
        atr   = float(df["ATR"].iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = float((df["high"].tail(10)-df["low"].tail(10)).mean())
        dist = max(atr * atr_mult, 0.1)

        if sinal == "COMPRA":
            stop   = entry - dist
            target = entry + dist * rr
        else:
            stop   = entry + dist
            target = entry - dist * rr
        return entry, stop, target, atr

    @staticmethod
    def avaliar_trade(mt5c, timeframe_name: str, detect_time: datetime,
                      entry: float, stop: float, target: float,
                      eval_bars: int = EVAL_BARS) -> tuple[str, datetime | None]:
        """Percorre os próximos N candles; retorna ('GAIN'/'LOSS'/'PENDING', datetime_ocorrencia)."""
        tf = TIMEFRAMES[timeframe_name]
        df_fw = mt5c.obter_candles_desde(tf, detect_time, eval_bars + 2)
        if df_fw.empty or len(df_fw) < 2:
            return "PENDING", None

        df_fw = df_fw.iloc[1:]  # ignora o candle de detecção
        for t, row in df_fw.iterrows():
            h = float(row["high"])
            l = float(row["low"])
            if target > entry:  # compra
                if l <= stop:   return "LOSS", t
                if h >= target: return "GAIN", t
            else:               # venda
                if h >= stop:   return "LOSS", t
                if l <= target: return "GAIN", t
        return "PENDING", df_fw.index[-1]

    # --- cole DENTRO da classe Analyzer ---

    @staticmethod
    def calcular_probabilidade_por_padrao(
            df: pd.DataFrame,
            padrao: str,
            rsi_buy: float = RSI_SOBREVENDIDO,
            rsi_sell: float = RSI_SOBRECOMPRA,
    ) -> tuple[int, str]:
        """
        Pontua qualquer padrão suportado com filtros de contexto (EMA/VWAP/RSI/Volume).
        Double Bottom/Top reutilizam a regra original (compatível com versões anteriores).
        Retorna (score 0..100, sinal 'COMPRA'|'VENDA'|'AGUARDAR').
        """
        # Se for DB/DT, usa a regra original:
        if padrao in ("Double Bottom", "Double Top"):
            return Analyzer.calcular_probabilidade(df, padrao, rsi_buy, rsi_sell)

        # Coleta contexto
        required = ("RSI", "EMA", "VWAP", "VOL_MA", "close", "tick_volume")
        assert all(c in df.columns for c in required), "Indicadores ausentes; chame Indicators.aplicar(df)."
        rsi = float(df["RSI"].iloc[-1])
        ema = float(df["EMA"].iloc[-1])
        vwap = float(df["VWAP"].iloc[-1])
        preco = float(df["close"].iloc[-1])
        vol = float(df["tick_volume"].iloc[-1])
        vol_ma = float(df["VOL_MA"].iloc[-1])

        score, sinal = 0, "AGUARDAR"

        # Reversões
        if padrao == "Head & Shoulders Top":
            if preco < ema:   score += 35
            if preco < vwap:  score += 25
            if rsi > 50:      score += 15
            if vol > vol_ma:  score += 25
            sinal = "VENDA"

        elif padrao == "Head & Shoulders Bottom":
            if preco > ema:   score += 35
            if preco > vwap:  score += 25
            if rsi < 55:      score += 15
            if vol > vol_ma:  score += 25
            sinal = "COMPRA"

        elif padrao == "Bullish Engulfing":
            if preco > ema:   score += 30
            if rsi > 50:      score += 30
            if vol > vol_ma:  score += 20
            if preco > vwap:  score += 20
            sinal = "COMPRA"

        elif padrao == "Bearish Engulfing":
            if preco < ema:   score += 30
            if rsi < 50:      score += 30
            if vol > vol_ma:  score += 20
            if preco < vwap:  score += 20
            sinal = "VENDA"

        elif padrao == "Hammer":
            if preco > ema:   score += 35
            if rsi > 45:      score += 25
            if vol > vol_ma:  score += 20
            if preco > vwap:  score += 20
            sinal = "COMPRA"

        elif padrao == "Shooting Star":
            if preco < ema:   score += 35
            if rsi < 55:      score += 25
            if vol > vol_ma:  score += 20
            if preco < vwap:  score += 20
            sinal = "VENDA"

        # Continuação/pausa
        elif padrao in ("Rectangle Bullish", "Triangle Bullish", "Flag Bullish", "Pennant Bullish"):
            if preco > ema:   score += 35
            if preco > vwap:  score += 25
            if vol > vol_ma:  score += 25
            if rsi > 50:      score += 15
            sinal = "COMPRA"

        elif padrao in ("Rectangle Bearish", "Triangle Bearish", "Flag Bearish", "Pennant Bearish"):
            if preco < ema:   score += 35
            if preco < vwap:  score += 25
            if vol > vol_ma:  score += 25
            if rsi < 50:      score += 15
            sinal = "VENDA"

        # clamp + opcional 'AGUARDAR' para scores fracos
        score = max(0, min(100, int(score)))
        if score < 50:
            return score, "AGUARDAR"
        return score, sinal
