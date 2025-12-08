# core/verdict_trader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
from datetime import datetime
import os

import MetaTrader5 as mt5

Side = Literal["BUY", "SELL"]


@dataclass
class VerdictTraderConfig:
    # Símbolo e execução
    symbol: str = os.getenv("VERDICT_SYMBOL", "XAUUSD")
    #lot: float = float(os.getenv("VERDICT_LOT", "0.02"))
    lot: float = float(os.getenv("VERDICT_LOT", "0.01"))
    deviation_points: int = int(os.getenv("VERDICT_DEVIATION_PTS", "60"))
    magic: int = int(os.getenv("VERDICT_MAGIC", "880000"))
    comment: str = os.getenv("VERDICT_COMMENT", "VerdictTrader")

    # Regras de decisão pelo score
    buy_threshold: float = float(os.getenv("VERDICT_BUY_THRESHOLD", "2.0"))     # >= 6.0 na sua config final
    sell_threshold: float = float(os.getenv("VERDICT_SELL_THRESHOLD", "-2.0"))  # <= -6.0 na sua config final

    # Stops (em POINTS, não em preço)
    sl_points: int = int(os.getenv("VERDICT_SL_POINTS", "450"))
    tp_points: int = int(os.getenv("VERDICT_TP_POINTS", "200"))
    trail_points: int = int(os.getenv("VERDICT_TRAIL_POINTS", "0"))  # 0 desabilita

    # Login opcional (se terminal já está logado, deixe vazio)
    login: Optional[int] = (int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN", "").isdigit() else None)
    password: Optional[str] = (os.getenv("MT5_PASSWORD") or None)
    server: Optional[str] = (os.getenv("MT5_SERVER") or None)

    # Caminho opcional do terminal (mt5.initialize(path))
    terminal_path: Optional[str] = (os.getenv("MT5_TERMINAL_PATH") or None)

    # Modos de preenchimento/tempo
    type_filling: int = int(os.getenv("VERDICT_TYPE_FILLING", str(mt5.ORDER_FILLING_FOK)))
    type_time: int = int(os.getenv("VERDICT_TYPE_TIME", str(mt5.ORDER_TIME_GTC)))

    # >>> NOVO: política de “apenas 1 operação por vez”
    # symbol = 1 posição por símbolo (padrão)
    # magic  = 1 posição por (símbolo + magic)
    # any    = 1 posição na conta inteira
    single_pos_scope: str = os.getenv("VERDICT_SINGLE_POS_SCOPE", "symbol").lower()


class VerdictTrader:
    """
    - Conecta ao MT5 (initialize + login opcional)
    - Decide o lado pelo 'score' e envia ordem MARKET com SL/TP em points
    - Suporte a trailing stop (chamar manage_trailing() no seu loop)
    - >>> Garante apenas 1 operação por vez, conforme single_pos_scope
    """

    def __init__(self, cfg: VerdictTraderConfig):
        self.cfg = cfg
        self._si = None  # cache de symbol_info

    # ---------- Conexão ----------
    def connect(self) -> bool:
        ok_init = mt5.initialize(self.cfg.terminal_path) if self.cfg.terminal_path else mt5.initialize()
        if not ok_init:
            print(f"[VerdictTrader] MT5.initialize() falhou: {mt5.last_error()}")
            return False

        if self.cfg.login and self.cfg.password and self.cfg.server:
            if not mt5.login(self.cfg.login, password=self.cfg.password, server=self.cfg.server):
                print(f"[VerdictTrader] MT5.login() falhou: {mt5.last_error()}")
                return False

        if not mt5.symbol_select(self.cfg.symbol, True):
            print(f"[VerdictTrader] symbol_select({self.cfg.symbol}) falhou.")
            return False

        self._si = mt5.symbol_info(self.cfg.symbol)
        if not self._si or self._si.point <= 0:
            print(f"[VerdictTrader] symbol_info inválido para {self.cfg.symbol}.")
            return False

        return True

    # ---------- Decisão pelo score ----------
    def decide_side(self, score: float) -> Optional[Side]:
        if score is None:
            return None
        if score >= self.cfg.buy_threshold:
            return "BUY"
        if score <= self.cfg.sell_threshold:
            return "SELL"
        return None

    # ---------- Guard de posição única ----------
    def _has_open_position(self) -> bool:
        scope = self.cfg.single_pos_scope
        if scope == "any":
            poss = mt5.positions_get()
            return bool(poss and len(poss) > 0)

        if scope == "symbol":
            poss = mt5.positions_get(symbol=self.cfg.symbol)
            return bool(poss and len(poss) > 0)

        if scope == "magic":
            poss = mt5.positions_get(symbol=self.cfg.symbol)
            if not poss:
                return False
            return any(getattr(p, "magic", None) == self.cfg.magic for p in poss)

        # fallback seguro
        poss = mt5.positions_get(symbol=self.cfg.symbol)
        return bool(poss and len(poss) > 0)

    # ---------- Utilidades ----------
    def _tick(self):
        return mt5.symbol_info_tick(self.cfg.symbol)

    def _point(self) -> float:
        return float(self._si.point)

    def _round(self, price: float) -> float:
        return float(round(price, self._si.digits))

    def _price_now(self, side: Side) -> Optional[float]:
        t = self._tick()
        if not t:
            return None
        return float(t.ask if side == "BUY" else t.bid)

    def _safe_levels(self, side: Side, entry_price: float) -> tuple[float, float]:
        pt = self._point()
        stops_level_pts = int(getattr(self._si, "stops_level", 0) or 0)

        sl_pts = max(self.cfg.sl_points, stops_level_pts) * pt
        tp_pts = max(self.cfg.tp_points, stops_level_pts) * pt

        if side == "BUY":
            sl = entry_price - sl_pts
            tp = entry_price + tp_pts
            if sl >= entry_price:
                sl = entry_price - max(pt, sl_pts)
        else:
            sl = entry_price + sl_pts
            tp = entry_price - tp_pts
            if sl <= entry_price:
                sl = entry_price + max(pt, sl_pts)

        return self._round(sl), self._round(tp)

    # ---------- Envio de ordem ----------
    def _order_send_market(self, side: Side, price_now: float, sl: float, tp: float) -> Dict[str, Any]:
        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": float(self.cfg.lot),
            "type": order_type,
            "price": float(price_now),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(self.cfg.deviation_points),
            "magic": int(self.cfg.magic),
            "comment": str(self.cfg.comment),
            "type_filling": int(self.cfg.type_filling),
            "type_time": int(self.cfg.type_time),
        }
        res = mt5.order_send(req)
        if res is None:
            return {"ok": False, "err": mt5.last_error(), "request": req}
        ok = (res.retcode == mt5.TRADE_RETCODE_DONE)
        return {"ok": ok, "retcode": res.retcode, "result": res, "request": req}

    # ---------- API principal ----------
    def decide_and_execute(self, score: float) -> Dict[str, Any]:
        # 1) Garante 1 operação por vez (conforme escopo configurado)
        if self._has_open_position():
            return {"ok": False, "reason": "Já existe posição aberta (regra de 1 operação por vez)."}

        # 2) Decide pelo score
        side = self.decide_side(score)
        if not side:
            return {"ok": False, "reason": f"Sem trade: score {score:.2f} fora dos thresholds."}

        # 3) Coleta preço e monta níveis
        px = self._price_now(side)
        if px is None:
            return {"ok": False, "reason": "Tick indisponível."}

        sl, tp = self._safe_levels(side, px)

        # 4) Envia
        sent = self._order_send_market(side, px, sl, tp)
        if not sent.get("ok"):
            return {"ok": False, "reason": "order_send falhou", **sent}

        info = {
            "ok": True,
            "side": side,
            "price": px,
            "sl": sl,
            "tp": tp,
            "time": datetime.utcnow().isoformat(),
            "ticket": getattr(sent["result"], "order", None),
        }
        print(f"[VerdictTrader] {side} @ {px} | SL {sl} | TP {tp} | ticket={info['ticket']}")
        return info

    # ---------- Trailing opcional ----------
    def manage_trailing(self) -> None:
        if self.cfg.trail_points <= 0:
            return

        pt = self._point()
        trail_dist = self.cfg.trail_points * pt
        t = self._tick()
        if not t:
            return

        # Para BUY usa BID; para SELL usa ASK
        price_now_buy = float(t.bid)
        price_now_sell = float(t.ask)

        poss = mt5.positions_get(symbol=self.cfg.symbol)
        if not poss:
            return

        for p in poss:
            if p.magic != self.cfg.magic:
                continue

            side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
            sl_old = float(p.sl) if p.sl else 0.0
            tp_old = float(p.tp) if p.tp else 0.0
            new_sl = None

            if side == "BUY":
                target_sl = price_now_buy - trail_dist
                if target_sl > 0 and (sl_old == 0.0 or target_sl > sl_old):
                    new_sl = self._round(target_sl)
            else:
                target_sl = price_now_sell + trail_dist
                if sl_old == 0.0 or target_sl < sl_old:
                    new_sl = self._round(target_sl)

            if new_sl is None:
                continue

            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "symbol": self.cfg.symbol,
                "sl": float(new_sl),
                "tp": float(tp_old),
                "magic": int(self.cfg.magic),
                "comment": f"{self.cfg.comment}-trail",
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[VerdictTrader] Trailing {side}: SL {sl_old} → {new_sl} (ticket {p.ticket})")
            else:
                print(f"[VerdictTrader] Falha trailing (ticket {p.ticket}):", res if res else mt5.last_error())
