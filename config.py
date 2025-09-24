"""Configurações globais do scanner XAUUSD."""

import MetaTrader5 as mt5

# === Chaves e Ativo ===
FRED_API_KEY = "c37a9221da8ffd2ad23c9f964882475b"
DEEPSEEK_API_KEY = "sk-6a50a8216f2e4db9a2e3698b2c505b06"
ATIVO = "XAUUSD"

# === DeepSeek endpoint (configurável) ===
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_PATH = "/v1/chat/completions"

# === Scanner / Indicadores ===
INTERVALO_MINUTOS = 5
QTD_CANDLES = 500
RSI_SOBRECOMPRA = 65
RSI_SOBREVENDIDO = 35
EMA_PERIOD = 21
TOLERANCIA_PADRAO = 0.0015

# === Avaliação/Trade Dinâmico ===
ATR_PERIOD = 14
ATR_MULT_STOP = 1.2
RR_MULT = 1.8
EVAL_BARS = 12

# === Auto-Tuning ===
TUNING_MIN_SIGNALS = 20
TUNING_INTERVAL_MIN = 60

# === Timeframes ===
TIMEFRAMES = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}
TF_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}

# === Consenso local ===
MIN_PROB_CONSENSO = 60
#PESOS_TIMEFRAMES = {"M5": 0.0, "M15": 1.5, "M30": 2.0, "H1": 2.5, "H4": 3.0}
PESOS_TIMEFRAMES = {"M5": 1.0, "M15": 1.5, "M30": 1.5, "H1": 1.5, "H4": 0.0} # Anterior bom
#PESOS_TIMEFRAMES = {"M5": 2.0, "M15": 0.5, "M30": 0.5, "H1": 0.5, "H4": 0.0} # A ser testado
#anteriormente o M5 era 1.0
MIN_SCORE_TRADE = 2.5  # só aciona sinal 'forte' se |score| >= 1.5

# === Padrões habilitados ===
ENABLED_PATTERNS = [
    "Head & Shoulders Top", "Head & Shoulders Bottom",
    "Double Top", "Double Bottom",
    "Bullish Engulfing", "Bearish Engulfing",
    "Hammer", "Shooting Star",
    "Rectangle Bullish", "Rectangle Bearish",
    "Triangle Bullish", "Triangle Bearish",
    "Flag Bullish", "Flag Bearish",
    "Pennant Bullish", "Pennant Bearish",
]

# === Parâmetros dos detectores ===
DETECTOR_PARAMS = {
    # Engulfing
    "ENGULFING_BODY_MIN_RATIO": 0.60,   # corpo atual >= 60% do corpo anterior
    # Pin Bar (Hammer/Shooting Star)
    "PINBAR_WICK_TO_BODY": 2.2,         # pavio maior que 2.2x corpo
    # Retângulo
    "RECT_LOOKBACK": 40,                 # barras para medir range horizontal
    "RECT_BAND_TOL": 0.003,              # 0.3% de tolerância na altura do retângulo
    # Triângulo
    "TRI_LOOKBACK": 40,                  # barras para linhas convergentes
    "TRI_MIN_CONV": 0.15,                # convergência mínima relativa
    # Flag / Pennant
    "FLAG_MIN_POLE_ATR": 2.0,            # flagpole >= 2x ATR médio
    "FLAG_MAX_BARS": 12,                 # duração máxima da bandeira/pennant
    # Confirmação
    "CONFIRM_BREAKOUT": True,            # exige candle fechar além da linha/neckline/banda
}


# ===================== LOGS & STORAGE =====================
# Caminhos dos CSVs usados pelo AutoTuner para registrar sinais e métricas
SIGNAL_LOG_CSV = "signals_log.csv"   # ex: caminho relativo ao projeto
PARAM_LOG_CSV  = "tuner_log.csv"     # ex: caminho relativo ao projeto



# no topo, com os demais imports
from core.trade_control import TradeControl, TradeControlConfig
