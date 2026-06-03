"""Configurações globais do scanner XAUUSD."""

import MetaTrader5 as mt5


# === Chaves e Ativo ===
FRED_API_KEY = "c37a9221da8ffd2ad23c9f964882475b"
# DEEPSEEK_API_KEY não é mais necessária ao usar Ollama local
# DEEPSEEK_API_KEY = "sk-..."
ATIVO = "XAUUSD"

# === Ollama endpoint (configurável) ===
# Quando integrado com Ollama local/substituto, configure o endpoint e o modelo abaixo.
# Exemplo: OLLAMA_BASE_URL = "http://109.199.107.136:11434/api/generate"
# e OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"
OLLAMA_BASE_URL = "http://109.199.107.136:11434/api/generate"
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"

# Mantemos as variáveis antigas apenas por compatibilidade (não utilizadas quando Ollama estiver ativo)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_PATH = "/v1/chat/completions"

# === Scanner / Indicadores ===
INTERVALO_MINUTOS = 5
QTD_CANDLES = 100
RSI_SOBRECOMPRA = 65
RSI_SOBREVENDIDO = 35
EMA_PERIOD = 21
TOLERANCIA_PADRAO = 0.0015

# === Avaliação/Trade Dinâmico ===
ATR_PERIOD = 14
ATR_MULT_STOP = 1.5
RR_MULT = 2.0
EVAL_BARS = 12

# === Verdict / Pre-IA filters (used by core.verdict_trader and main prechecks)
# Habilita checagem multi-timeframe se a EMA estiver 'flat' para economizar chamadas à IA
VERDICT_FLAT_FILTER_ENABLE = True
# Timeframes a checar para o filtro de EMA "flat" (chaves do dicionário TIMEFRAMES/TF_MINUTES)
VERDICT_FLAT_TFS = ["M15", "M30"]
# Modo de checagem: 'atr' (delta EMA < k * ATR), 'abs' (valor absoluto), ou 'pct' (fração)
VERDICT_FLAT_MODE = "atr"
# Lookback (n barras) usado para comparar EMA atual vs EMA passada
VERDICT_FLAT_LOOKBACK = 24
# Coeficiente k aplicado ao ATR para definir limiar de 'flat'
VERDICT_FLAT_ATR_K = 0.5

# Habilita filtro de proximidade (preço dentro de k * ATR da EMA9) antes da chamada à IA
VERDICT_PROXIMITY_FILTER_ENABLE = True
# k usado no cálculo de proximidade em relação à ATR (ema9 ± k * ATR)
VERDICT_PROXIMITY_ATR_K = 1.0

# Fonte do veredito final: 'deepseek' = exigir decisão da IA;
# 'local' = não chamar a IA e seguir veredito local;
# 'hybrid' = (padrão) usar DeepSeek quando disponível, senão fallback local.
VERDICT_FINAL_SOURCE = "local"

# === Auto-Tuning ===
TUNING_MIN_SIGNALS = 20
TUNING_INTERVAL_MIN = 60

# === Timeframes ===
TIMEFRAMES = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}
TF_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}

# === Consenso local ===
MIN_PROB_CONSENSO = 60
#PESOS_TIMEFRAMES = {"M5": 0.0, "M15": 1.5, "M30": 2.0, "H1": 2.5, "H4": 3.0}
#PESOS_TIMEFRAMES = {"M5": 1.0, "M15": 1.5, "M30": 1.5, "H1": 1.5, "H4": 0.0} # Anterior bom
PESOS_TIMEFRAMES = {"M5": 0.0, "M15": 2.0, "M30": 2.0, "H1": 0.0, "H4": 0.0}  # A ser testado
#anteriormente o M5 era 1.0
MIN_SCORE_TRADE = 2.0  # só aciona sinal 'forte' se |score| >= 1.5

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
