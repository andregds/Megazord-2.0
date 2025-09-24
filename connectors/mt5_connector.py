"""MT5Connector
Responsável por conectar ao MetaTrader 5 e fornecer velas/candles.
Uso:
    mt5c = MT5Connector(symbol="XAUUSD")
    mt5c.conectar()
    df = mt5c.obter_candles(mt5.TIMEFRAME_M5, count=500)
"""

import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
from config import ATIVO, QTD_CANDLES

class MT5Connector:
    """Encapsula operações de conexão e leitura de dados do MT5."""

    def __init__(self, symbol: str = ATIVO):
        """Inicializa com o símbolo (ativo) desejado."""
        self.symbol = symbol

    def conectar(self) -> bool:
        """Inicializa MT5 e seleciona o símbolo para uso."""
        if not mt5.initialize():
            raise RuntimeError(f"Erro ao conectar MT5: {mt5.last_error()}")
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"Erro ao selecionar ativo {self.symbol}")
        return True

    def obter_candles(self, timeframe, count: int = QTD_CANDLES) -> pd.DataFrame:
        """Retorna um DataFrame com os últimos `count` candles do timeframe."""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, int(count))
        df = pd.DataFrame(rates)
        if df.empty:
            return pd.DataFrame()
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    def obter_candles_desde(self, timeframe, dt_from: datetime, count: int) -> pd.DataFrame:
        """Retorna candles a partir de `dt_from` (para avaliar sinais pendentes)."""
        rates = mt5.copy_rates_from(self.symbol, timeframe, dt_from, int(count))
        df = pd.DataFrame(rates)
        if df.empty:
            return pd.DataFrame()
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df
