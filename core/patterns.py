
"""PatternDetector
Detecta padrões gráficos com heurísticas práticas e baratas:
- Head & Shoulders (Top/Bottom) com confirmação do neckline
- Double Top / Double Bottom
- Engulfing (bullish/bearish)
- Pin Bar (Hammer / Shooting Star)
- Rectangle (rompido para cima/baixo)
- Triangle (rompido; linhas convergentes)
- Flag / Pennant (flagpole + consolidação curta)

Use junto com filtros de contexto (EMA/VWAP/RSI/Volume) no Analyzer.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from config import TOLERANCIA_PADRAO, DETECTOR_PARAMS

class PatternDetector:
 """Regras de detecção de padrões (expansível)."""

 SUPPORTED = [
     "Head & Shoulders Top", "Head & Shoulders Bottom",
     "Double Top", "Double Bottom",
     "Bullish Engulfing", "Bearish Engulfing",
     "Hammer", "Shooting Star",
     "Rectangle Bullish", "Rectangle Bearish",
     "Triangle Bullish", "Triangle Bearish",
     "Flag Bullish", "Flag Bearish",
     "Pennant Bullish", "Pennant Bearish",
 ]

 @staticmethod
 def listar_padroes() -> List[str]:
     return PatternDetector.SUPPORTED.copy()

 # ---------- Candles simples ----------

 @staticmethod
 def engulfing(df: pd.DataFrame, body_ratio: float) -> Optional[str]:
     o, c = df["open"].values, df["close"].values
     if len(o) < 2:
         return None
     prev_body = abs(c[-2] - o[-2]); curr_body = abs(c[-1] - o[-1])
     if curr_body < max(prev_body * body_ratio, 1e-9):
         return None
     # Bullish
     if c[-2] < o[-2] and c[-1] > o[-1] and (o[-1] <= min(o[-2], c[-2])) and (c[-1] >= max(o[-2], c[-2])):
         return "Bullish Engulfing"
     # Bearish
     if c[-2] > o[-2] and c[-1] < o[-1] and (o[-1] >= min(o[-2], c[-2])) and (c[-1] <= max(o[-2], c[-2])):
         return "Bearish Engulfing"
     return None

 @staticmethod
 def pin_bar(df: pd.DataFrame, wick_to_body: float) -> Optional[str]:
     o, c = df["open"].iloc[-1], df["close"].iloc[-1]
     h, l = df["high"].iloc[-1], df["low"].iloc[-1]
     body = abs(c - o)
     upper = h - max(o, c)
     lower = min(o, c) - l
     if body == 0:
         return None
     if lower / body >= wick_to_body and upper <= body * 0.5:
         return "Hammer"
     if upper / body >= wick_to_body and lower <= body * 0.5:
         return "Shooting Star"
     return None

 # ---------- Reversão clássica ----------

 @staticmethod
 def double_top_bottom(df: pd.DataFrame, tolerancia: float) -> Optional[str]:
     closes = df["close"].values
     highs = df["high"].values
     lows  = df["low"].values
     if len(closes) < 4:
         return None
     if abs(lows[-1] - lows[-3]) / max(1e-9, lows[-1]) < tolerancia and closes[-1] > df["EMA"].iloc[-1]:
         return "Double Bottom"
     if abs(highs[-1] - highs[-3]) / max(1e-9, highs[-1]) < tolerancia and closes[-1] < df["EMA"].iloc[-1]:
         return "Double Top"
     return None

 @staticmethod
 def head_shoulders(df: pd.DataFrame, confirm_breakout: bool) -> Optional[str]:
     n = min(len(df), 50)
     if n < 15:
         return None
     highs = df["high"].tail(n)
     lows  = df["low"].tail(n)

     def pivots(series, look=2, mode="top"):
         idx, vals = [], []
         arr, idxs = series.values, series.index
         for i in range(look, len(arr)-look):
             win = arr[i-look:i+look+1]
             if mode == "top" and arr[i] == win.max() and (arr[i] > win[0]) and (arr[i] > win[-1]):
                 idx.append(idxs[i]); vals.append(arr[i])
             if mode == "bot" and arr[i] == win.min() and (arr[i] < win[0]) and (arr[i] < win[-1]):
                 idx.append(idxs[i]); vals.append(arr[i])
         return idx, vals

     ti, tv = pivots(highs, 2, "top")
     bi, bv = pivots(lows, 2, "bot")
     if len(tv) < 3 or len(bv) < 2:
         return None

     # H&S Top
     if len(tv) >= 3:
         L, H, R = tv[-3], tv[-2], tv[-1]
         if H > L and H > R and 0.8 <= (L / R) <= 1.2:
             if len(bv) >= 2:
                 nl = (bv[-2] + bv[-1]) / 2.0
                 if (not confirm_breakout) or (df["close"].iloc[-1] < nl):
                     return "Head & Shoulders Top"

     # H&S Bottom
     if len(bv) >= 3:
         L, Hh, R = bv[-3], bv[-2], bv[-1]
         if Hh < L and Hh < R and 0.8 <= (L / R) <= 1.2:
             if len(tv) >= 2:
                 nl = (tv[-2] + tv[-1]) / 2.0
                 if (not confirm_breakout) or (df["close"].iloc[-1] > nl):
                     return "Head & Shoulders Bottom"

     return None

 # ---------- Continuação / pausa ----------

 @staticmethod
 def rectangle(df: pd.DataFrame, lookback: int, band_tol: float, confirm_breakout: bool) -> Optional[str]:
     if len(df) < lookback:
         return None
     win = df.tail(lookback)
     top = win["high"].max()
     bot = win["low"].min()
     mid = (top + bot) / 2.0
     height = (top - bot) / max(1e-9, mid)
     if height > band_tol:
         return None
     close = df["close"].iloc[-1]
     if close > top and ((not confirm_breakout) or (df["close"].iloc[-2] <= top)):
         return "Rectangle Bullish"
     if close < bot and ((not confirm_breakout) or (df["close"].iloc[-2] >= bot)):
         return "Rectangle Bearish"
     return None

 @staticmethod
 def triangle(df: pd.DataFrame, lookback: int, min_conv: float, confirm_breakout: bool) -> Optional[str]:
     if len(df) < lookback:
         return None
     win = df.tail(lookback)
     upper = pd.Series(win["high"]).rolling(3).max().dropna()
     lower = pd.Series(win["low"]).rolling(3).min().dropna()
     if len(upper) < 5 or len(lower) < 5:
         return None
     su, iu = np.polyfit(np.arange(len(upper)), upper.values, 1)
     sl, il = np.polyfit(np.arange(len(lower)), lower.values, 1)
     if su >= 0 or sl <= 0:
         return None
     dist0 = (upper.iloc[0] - lower.iloc[0]) / max(1e-9, win["close"].iloc[0])
     dist1 = (upper.iloc[-1] - lower.iloc[-1]) / max(1e-9, win["close"].iloc[-1])
     if (dist0 - dist1) < min_conv * max(dist0, 1e-9):
         return None
     last = df.iloc[-1]
     u_last = su * (len(upper)-1) + iu
     l_last = sl * (len(lower)-1) + il
     if last["close"] > u_last and ((not confirm_breakout) or df["close"].iloc[-2] <= u_last):
         return "Triangle Bullish"
     if last["close"] < l_last and ((not confirm_breakout) or df["close"].iloc[-2] >= l_last):
         return "Triangle Bearish"
     return None

 @staticmethod
 def flag_or_pennant(df: pd.DataFrame, max_bars: int, min_pole_atr: float, confirm_breakout: bool) -> Optional[str]:
     if len(df) < max_bars + 10:
         return None
     atr = df["ATR"].tail(20).mean()
     if np.isnan(atr) or atr <= 0:
         return None
     tail = df.tail(max_bars + 8)
     move = tail["close"].iloc[-1] - tail["close"].iloc[-(max_bars+8)]
     direction = "up" if move > 0 else "down"
     if abs(move) < min_pole_atr * atr:
         return None
     block = df.tail(max_bars)
     if (block["high"].max() - block["low"].min()) > 1.2 * abs(move)/2:
         return None
     hi = block["high"].max(); lo = block["low"].min()
     c = df["close"].iloc[-1]
     if direction == "up" and c > hi and ((not confirm_breakout) or df["close"].iloc[-2] <= hi):
         rng = block["high"] - block["low"]
         return "Pennant Bullish" if rng.std() < rng.mean()*0.5 else "Flag Bullish"
     if direction == "down" and c < lo and ((not confirm_breakout) or df["close"].iloc[-2] >= lo):
         rng = block["high"] - block["low"]
         return "Pennant Bearish" if rng.std() < rng.mean()*0.5 else "Flag Bearish"
     return None

 # ---------- Orquestrador ----------


 @staticmethod
 def detect_all(df: pd.DataFrame, tolerancia: float, params: dict) -> List[str]:
     out: List[str] = []

     # Remove a vela atual (em formação) — usa apenas velas fechadas
     df = df.iloc[:-1].copy()

     p = PatternDetector.double_top_bottom(df, tolerancia)
     if p: out.append(p)

     p = PatternDetector.head_shoulders(df, confirm_breakout=params.get("CONFIRM_BREAKOUT", True))
     if p: out.append(p)

     p = PatternDetector.engulfing(df, body_ratio=params.get("ENGULFING_BODY_MIN_RATIO", 0.6))
     if p: out.append(p)

     p = PatternDetector.pin_bar(df, wick_to_body=params.get("PINBAR_WICK_TO_BODY", 2.2))
     if p: out.append(p)

     p = PatternDetector.rectangle(df, params.get("RECT_LOOKBACK", 40), params.get("RECT_BAND_TOL", 0.003), params.get("CONFIRM_BREAKOUT", True))
     if p: out.append(p)

     p = PatternDetector.triangle(df, params.get("TRI_LOOKBACK", 40), params.get("TRI_MIN_CONV", 0.15), params.get("CONFIRM_BREAKOUT", True))
     if p: out.append(p)

     p = PatternDetector.flag_or_pennant(df, params.get("FLAG_MAX_BARS", 12), params.get("FLAG_MIN_POLE_ATR", 2.0), params.get("CONFIRM_BREAKOUT", True))
     if p: out.append(p)

     from config import ENABLED_PATTERNS  # import tardio evita ciclos
     return [x for x in out if x in ENABLED_PATTERNS]

__all__ = ["PatternDetector"]
