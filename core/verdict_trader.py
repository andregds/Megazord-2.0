# FILE NAME: core/verdict_trader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
from datetime import datetime
import os
import MetaTrader5 as mt5
import logging
import pandas as pd
from core.indicators import Indicators # Importação confirmada
from config import VERDICT_FLAT_FILTER_ENABLE, VERDICT_FLAT_MODE, VERDICT_FLAT_LOOKBACK, VERDICT_FLAT_ATR_K, VERDICT_PROXIMITY_ATR_K

# Importar o novo módulo DailyProfitChecker
from core.daily_profit_checker import DailyProfitChecker, PROFIT_TARGET # Importamos também PROFIT_TARGET para logar

# Configura o logger da aplicação (se já não estiver configurado em main.py)
logger = logging.getLogger("Megazord")
if not logger.handlers: # Configura apenas se não houver handlers, para evitar duplicação
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

Side = Literal["BUY", "SELL"]

@dataclass
class VerdictTraderConfig:
    # Símbolo e execução
    symbol: str = os.getenv("VERDICT_SYMBOL", "BTCUSD")
    lot: float = float(os.getenv("VERDICT_LOT", "0.02"))
    deviation_points: int = int(os.getenv("VERDICT_DEVIATION_PTS", "60"))
    magic: int = int(os.getenv("VERDICT_MAGIC", "880031"))
    comment: str = os.getenv("VERDICT_COMMENT", "VerdictTrader")
    # Regras de decisão pelo score
    buy_threshold: float = float(os.getenv("VERDICT_BUY_THRESHOLD", "2.0"))
    sell_threshold: float = float(os.getenv("VERDICT_SELL_THRESHOLD", "-2.0"))
    # Parâmetros para cálculo de SL/TP (se não vierem da IA)
    atr_period: int = int(os.getenv("VERDICT_ATR_PERIOD", "14"))
    atr_multiplier_sl: float = float(os.getenv("VERDICT_ATR_MULTIPLIER_SL", "1.5"))
    atr_multiplier_tp: float = float(os.getenv("VERDICT_ATR_MULTIPLIER_TP", "3.0"))
    # Parâmetros para validação local
    rsi_period: int = int(os.getenv("VERDICT_RSI_PERIOD", "14"))
    rsi_overbought: int = int(os.getenv("VERDICT_RSI_OVERBOUGHT", "70"))
    rsi_oversold: int = int(os.getenv("VERDICT_RSI_OVERSOLD", "30"))
    ema_period: int = int(os.getenv("VERDICT_EMA_PERIOD", "21"))

class VerdictTrader:
    def __init__(self, config: VerdictTraderConfig):
        self.cfg = config
        self.symbol_info = None
        self.point = 0.0
        self.mt5_connected = False
        # Inicializa o DailyProfitChecker
        self.daily_profit_checker = DailyProfitChecker(symbol=self.cfg.symbol)

    def connect(self) -> bool:
        if not mt5.initialize():
            logger.error(f"Falha ao inicializar MT5: {mt5.last_error()}")
            return False
        # Conectar à conta (se credenciais forem fornecidas)
        mt5_login = os.getenv("MT5_LOGIN")
        mt5_password = os.getenv("MT5_PASSWORD")
        mt5_server = os.getenv("MT5_SERVER")
        if mt5_login and mt5_password and mt5_server:
            login = int(mt5_login)
            if not mt5.login(login, password=mt5_password, server=mt5_server):
                logger.error(f"Falha ao conectar à conta MT5 {login} no servidor {mt5_server}: {mt5.last_error()}")
                mt5.shutdown()
                return False
            logger.info(f"Conectado ao MetaTrader5 na conta {login}.")
        else:
            logger.info("MetaTrader5 inicializado (sem login explícito, usando conta padrão).")
        # Obter informações do símbolo
        self.symbol_info = mt5.symbol_info(self.cfg.symbol)
        if self.symbol_info is None:
            logger.error(f"Falha ao obter informações do símbolo {self.cfg.symbol}")
            mt5.shutdown()
            return False
        if not self.symbol_info.visible:
            if not mt5.symbol_select(self.cfg.symbol, True):
                logger.error(f"Falha ao selecionar o símbolo {self.cfg.symbol}")
                mt5.shutdown()
                return False
        self.point = self.symbol_info.point
        logger.info(f"Símbolo {self.cfg.symbol} selecionado e informações obtidas. Point: {self.point}")
        self.mt5_connected = True
        return True

    def _get_current_price(self, side: Side) -> Optional[float]:
        """Obtém o preço atual (bid para SELL, ask para BUY)."""
        if not self.mt5_connected:
            logger.error("MT5 não conectado. Não é possível obter o preço atual.")
            return None
        symbol_info_tick = mt5.symbol_info_tick(self.cfg.symbol)
        if symbol_info_tick is None:
            logger.error(f"Falha ao obter tick para {self.cfg.symbol}: {mt5.last_error()}")
            return None
        if side == "BUY":
            return symbol_info_tick.ask
        elif side == "SELL":
            return symbol_info_tick.bid
        return None

    # --- Proteções e checagens utilitárias ---
    def is_market_flat(self, df: Optional[pd.DataFrame], mode: str = None, lookback: int = None) -> bool:
        """Detecta se a EMA está 'flat' no dataframe fornecido.
        Modo 'atr' compara delta ema versus k * ATR; retorna True se flat (i.e., bloquear).
        """
        try:
            if df is None or df.empty:
                return True
            if mode is None:
                mode = VERDICT_FLAT_MODE
            if lookback is None:
                lookback = VERDICT_FLAT_LOOKBACK
            if len(df) < lookback + 2:
                return True
            if 'EMA' not in df.columns:
                return True
            ema_now = float(df['EMA'].iloc[-1])
            ema_past = float(df['EMA'].iloc[-(lookback+1)])
            delta = abs(ema_now - ema_past)
            if mode == 'atr':
                if 'ATR' not in df.columns:
                    return True
                atr_now = float(df['ATR'].iloc[-1])
                thr = float(VERDICT_FLAT_ATR_K) * atr_now
                return delta < thr
            elif mode == 'abs':
                # fallback absolute threshold (0.5 price units)
                return delta < 0.5
            else: # pct
                return (delta / max(abs(ema_past), 1.0)) < 0.001
        except Exception:
            return True

    def is_price_within_proximity(self, df: Optional[pd.DataFrame], price: float, k: Optional[float] = None) -> bool:
        """Verifica se o preço está dentro de k * ATR da EMA9. Retorna True se OK (permitir operar)."""
        try:
            if df is None or df.empty:
                return False
            if k is None:
                k = VERDICT_PROXIMITY_ATR_K
            if 'ema_9' not in df.columns or 'ATR' not in df.columns:
                return False
            ema9 = float(df['ema_9'].iloc[-1])
            atr = float(df['ATR'].iloc[-1])
            lim = float(k) * atr
            return abs(price - ema9) <= lim
        except Exception:
            return False

    # Renomeado para _price_now para consistência com a discussão anterior
    _price_now = _get_current_price

    def validar_decisao_local(self, side: Side, price: float) -> None:
        """
        Valida a decisão de trade (BUY/SELL) contra indicadores técnicos locais (RSI, EMA).
        Esta função apenas loga alertas, não bloqueia a execução da ordem.
        """
        if not self.mt5_connected:
            logger.warning("MT5 não conectado. Validação local de decisão não pode ser realizada.")
            return
        # Obter os últimos candles para cálculo dos indicadores
        # Usamos um timeframe menor (M5) para validação rápida, ou o menor timeframe configurado
        # Assumindo que o MT5Connector tem um método para obter candles para um timeframe específico
        # Para esta validação, 100 candles devem ser suficientes para RSI e EMA
        candles = mt5.copy_rates_from_pos(self.cfg.symbol, mt5.TIMEFRAME_M5, 0, 100)
        if candles is None or len(candles) == 0:
            logger.warning("Não foi possível obter candles para validação local. Validação ignorada.")
            return
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        # Aplicar indicadores usando o método 'aplicar' da classe Indicators
        # Passamos os períodos configurados para RSI e EMA
        df = Indicators.aplicar(df, rsi_period=self.cfg.rsi_period, ema_period=self.cfg.ema_period)
        # Verificar se as colunas 'RSI' e 'EMA' foram adicionadas e não estão vazias
        if 'RSI' in df.columns and not df['RSI'].empty:
            current_rsi = df['RSI'].iloc[-1]
            if side == "BUY" and current_rsi > self.cfg.rsi_overbought:
                logger.warning(
                    f"⚠️ Validação Local: Decisão de COMPRA com RSI ({current_rsi:.2f}) em sobrecompra (> {self.cfg.rsi_overbought})."
                )
            elif side == "SELL" and current_rsi < self.cfg.rsi_oversold:
                logger.warning(
                    f"⚠️ Validação Local: Decisão de VENDA com RSI ({current_rsi:.2f}) em sobrevenda (< {self.cfg.rsi_oversold})."
                )
        else:
            logger.warning("⚠️ Validação Local: RSI não calculado ou coluna 'RSI' ausente após aplicar indicadores.")
        if 'EMA' in df.columns and not df['EMA'].empty:
            current_ema = df['EMA'].iloc[-1]
            if side == "BUY" and price < current_ema:
                logger.warning(
                    f"⚠️ Validação Local: Decisão de COMPRA com preço ({price:.5f}) abaixo da EMA ({current_ema:.5f})."
                )
            elif side == "SELL" and price > current_ema:
                logger.warning(
                    f"⚠️ Validação Local: Decisão de VENDA com preço ({price:.5f}) acima da EMA ({current_ema:.5f})."
                )
        else:
            logger.warning("⚠️ Validação Local: EMA não calculado ou coluna 'EMA' ausente após aplicar indicadores.")

    def _order_send_market(self, side: Side, price: float, sl: float, tp: float) -> Dict[str, Any]:
        """Envia uma ordem a mercado com SL e TP."""
        if not self.mt5_connected:
            logger.error("MT5 não conectado. Ordem não enviada.")
            return {"ok": False, "comment": "MT5 not connected"}

        # --- NOVO CÓDIGO AQUI: Checagem de limite de lucro diário ANTES de enviar a ordem ---
        if self.daily_profit_checker.update_and_check_profit():
            logger.info(f"Operações bloqueadas para {self.cfg.symbol}: Limite de lucro diário de ${PROFIT_TARGET:.2f} atingido.")
            return {"ok": False, "comment": "Daily profit limit reached"}
        # --- FIM DO NOVO CÓDIGO ---

        # Validação local da decisão (apenas para logar alertas, não bloqueia)
        self.validar_decisao_local(side, price) # Chama para logar alertas, mas não bloqueia
        # Prepara a requisição
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": self.cfg.lot,
            "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": self.cfg.deviation_points,
            "sl": sl,
            "tp": tp,
            "magic": self.cfg.magic,
            "comment": self.cfg.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Ordem {side} enviada com sucesso. Ticket: {result.order}")
            return {"ok": True, "result": result}
        else:
            logger.error(f"Falha ao enviar ordem {side}: {result.comment} (Retcode: {result.retcode})")
            return {"ok": False, "retcode": result.retcode, "comment": result.comment}

    def manage_trailing(self, side: str, price: float, trail_distance: float):
        """
        Gerencia o trailing stop para posições abertas.
        Ajusta o SL para proteger lucros, mantendo o TP original.
        """
        if not self.mt5_connected:
            logger.error("MT5 não conectado. Trailing stop não gerenciado.")
            return
        positions = mt5.positions_get(symbol=self.cfg.symbol)
        if positions is None:
            logger.error(f"Falha ao obter posições para {self.cfg.symbol}: {mt5.last_error()}")
            return
        for p in positions:
            if p.magic != self.cfg.magic:
                continue # Ignora posições de outros robôs/operações
            # Garante que estamos gerenciando a posição correta (BUY/SELL)
            if (side == "BUY" and p.type == mt5.ORDER_TYPE_BUY) or \
               (side == "SELL" and p.type == mt5.ORDER_TYPE_SELL):
               #(side == "BUY" and p.type == mt5.ORDER_TYPE_BUY) or \   CORRETO
               #(side == "SELL" and p.type == mt5.ORDER_TYPE_SELL):     CORRETO

                # Calcula o novo SL
                new_sl = p.sl # Começa com o SL atual
                if side == "BUY":
                    # Se o preço atual subiu o suficiente para mover o SL
                    if price - trail_distance > p.sl:
                        new_sl = price - trail_distance
                else: # SELL
                    # Se o preço atual caiu o suficiente para mover o SL
                    if price + trail_distance < p.sl or p.sl == 0.0: # Considera SL=0 como não definido
                        new_sl = price + trail_distance
                # Arredonda o novo SL para o número de dígitos do símbolo
                new_sl = round(new_sl, self.symbol_info.digits)
                # Se o novo SL for diferente do SL atual e for mais favorável (protegendo mais lucro)
                if new_sl != p.sl and \
                   ((side == "BUY" and new_sl > p.sl) or (side == "SELL" and new_sl < p.sl)):
                    # Prepara a requisição de modificação
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": p.ticket,
                        "sl": float(new_sl),
                        "tp": float(p.tp), # Mantém o TP original definido pela IA
                        "magic": int(self.cfg.magic),
                        "comment": f"{self.cfg.comment}-trail",
                    }
                    res = mt5.order_send(req)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"[VerdictTrader] Trailing {side}: SL {p.sl} → {new_sl} (ticket {p.ticket})")
                    else:
                        logger.error(f"[VerdictTrader] Falha trailing (ticket {p.ticket}): {res if res else mt5.last_error()}")

