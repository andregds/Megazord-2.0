# FILE NAME: core/trade_control.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, List
from datetime import datetime, timedelta

import MetaTrader5 as mt5

Side = Literal["BUY", "SELL"]

@dataclass
class TradeControlConfig:
    symbol: str
    magic: int

    # Lote base e martingale
    base_lot: float = 0.25  #Não funciona
    martingale_factor: float = 1.0 #Não funciona
    max_lot: float = 99.60  # trava de segurança

    # Janela do histórico para procurar o último resultado
    history_days: int = 1

    # Disciplina: apenas 1 posição por vez?
    one_position_only: bool = True

class TradeControl:
    """
    Controle simples:
      - 1 posição por vez (símbolo + magic)
      - Martingale baseado em perdas consecutivas (sem cooldown)
    """

    def __init__(self, cfg: TradeControlConfig):
        self.cfg = cfg
        self._has_open_position: bool = False  # atualizado no sync()
        self._last_sent: dict = {}             # opcional: log informativo

    # ------------------------- Posições -------------------------

    def _open_positions(self) -> List:
        """Retorna posições abertas do símbolo e magic desta estratégia."""
        poss = mt5.positions_get(symbol=self.cfg.symbol) or []
        out = []
        for p in poss:
            try:
                if getattr(p, "magic", 0) != self.cfg.magic:
                    continue
                out.append(p)
            except Exception:
                continue
        return out

    def sync(self) -> None:
        """Atualiza flag interna se há posição aberta."""
        self._has_open_position = len(self._open_positions()) > 0

    def is_position_open(self) -> bool:
        """True se já existe posição aberta (símbolo+magic)."""
        return bool(self._has_open_position)

    # --------------------- Histórico / perdas ---------------------

    def _closed_deals(self):
        """Deals fechados (DEAL_ENTRY_OUT) filtrados por symbol/magic."""
        end = datetime.utcnow()
        start = end - timedelta(days=self.cfg.history_days)
        deals = mt5.history_deals_get(start, end) or []
        out = []
        for d in deals:
            try:
                if getattr(d, "symbol", "") != self.cfg.symbol:
                    continue
                if getattr(d, "magic", 0) != self.cfg.magic:
                    continue
                if getattr(d, "entry", None) != mt5.DEAL_ENTRY_OUT:
                    continue
                out.append(d)
            except Exception:
                continue
        # mais recente primeiro
        out.sort(key=lambda x: getattr(x, "time_msc", getattr(x, "time", 0)), reverse=True)
        return out

    def consecutive_losses(self) -> int:
        """Conta perdas consecutivas até o último gain (ou início da janela)."""
        losses = 0
        for d in self._closed_deals():
            p = float(getattr(d, "profit", 0.0))
            if p <= 0:
                losses += 1
            else:
                break
        return losses

    # --------------------- Martingale ---------------------

    def next_lot(self, base: Optional[float] = None) -> float:
        """
        (base ou cfg.base_lot) * (martingale_factor ** consecutive_losses), limitado por max_lot.
        """
        losses = self.consecutive_losses()
        base_lot = float(base) if (base is not None) else float(self.cfg.base_lot)
        lot = base_lot * (self.cfg.martingale_factor ** max(0, losses))
        return min(float(self.cfg.max_lot), float(lot))

    # Alias usado no main
    def get_next_lot(self) -> float:
        return self.next_lot()

    def on_order_sent(self, ticket: Optional[int], lot: float, side: Optional[Side]) -> None:
        """Hook opcional para log."""
        self._last_sent = {"ticket": ticket, "lot": float(lot), "side": side}
