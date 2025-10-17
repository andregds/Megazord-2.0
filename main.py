"""main
Ponto de entrada: orquestra o scanner multi-timeframe, auto-tuning, consenso local
e veredito do DeepSeek (com fallback). Agora suporta múltiplos padrões clássicos.

Execute:
    python main.py
"""

import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta

from config import (
    DEEPSEEK_API_KEY,
    INTERVALO_MINUTOS, TIMEFRAMES, QTD_CANDLES,
    MIN_PROB_CONSENSO, PESOS_TIMEFRAMES,
    MIN_SCORE_TRADE, DETECTOR_PARAMS,
)

from connectors.mt5_connector import MT5Connector
from core.indicators import Indicators
from core.patterns import PatternDetector
from core.analyzer import Analyzer
from core.deepseek_client import DeepSeekClient
from core.autotuner import AutoTuner
from core.verdict import VerdictAggregator
from core.signal_notifier import SignalNotifier
from trader_ai import TraderAI

# Executor de ordens baseado no score (BUY/SELL)
from core.verdict_trader import VerdictTrader, VerdictTraderConfig

# Controle: martingale + 1 posição por vez (SEM cooldown interno do TradeControl)
from core.trade_control import TradeControl, TradeControlConfig


# --- Credenciais do Telegram (via ENV, com fallback) ---
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "8203615899:AAGhBysgD88tEvZdyXvI-hL6IU6WVxW7950")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "183379814")

# Ponte: algumas libs leem a chave do ambiente; garantimos aqui
os.environ.setdefault("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
os.environ.setdefault("TELEGRAM_TOKEN", TELEGRAM_TOKEN)
os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)

# Mostrar no terminal o que está sendo procurado
print("🔧 Padrões buscados:", ", ".join(PatternDetector.listar_padroes()))


class Monitor:
    """Loop principal: coleta dados, detecta padrões, calcula prob/sinal, loga e decide veredito."""

    def __init__(self):
        """Inicializa conectores e componentes principais."""
        self.mt5 = MT5Connector()
        self.deepseek = DeepSeekClient()
        self.tuner = AutoTuner()
        self.historico_padroes = []

        # Notificador (envia ao Telegram quando regras forem atendidas)
        self.notifier = SignalNotifier(
            token=TELEGRAM_TOKEN,
            chat_id=TELEGRAM_CHAT_ID,
            required_tfs=list(TIMEFRAMES.keys()),
            min_score=6.5,
            cooldown_sec=180,
        )

        # === Execução automática pelo score (BUY/SELL) ===
        self.vt = VerdictTrader(VerdictTraderConfig())
        ok_conn = self.vt.connect()
        if ok_conn:
            print("[VerdictTrader] ✅ Conectado ao MT5 e símbolo preparado.")
        else:
            print("[VerdictTrader] ❌ Falha de conexão/login no MT5. "
                  "(Se o terminal já estiver logado manualmente, deixe MT5_LOGIN/MT5_PASSWORD/MT5_SERVER vazios.)")

        # ---------- Adaptador de envio de ordens p/ o TraderAI (opcional) ----------
        def _ai_send_order_wrapper(trader, side, px_now, entry, sl, tp, lots=None, near_pct=0.0001):
            try:
                lots = lots or 0.01
                # MARKET se entry muito perto do preço atual
                if max(px_now, 1) > 0 and entry is not None and abs(entry - px_now) / px_now < float(near_pct):
                    fn = getattr(self.mt5, "market_order", None) or getattr(self.mt5, "enviar_ordem_market", None)
                    if callable(fn):
                        return fn(side=side, lots=lots, sl=sl, tp=tp)
                    print(f"[SIM] MARKET {side} lots={lots} sl={sl} tp={tp}")
                    return {"status": "simulado", "type": "MARKET", "side": side, "lots": lots, "sl": sl, "tp": tp}

                # Decide tipo da pendente
                if side.upper() == "BUY":
                    kind = "BUY_LIMIT" if entry < px_now else "BUY_STOP"
                else:
                    kind = "SELL_LIMIT" if entry > px_now else "SELL_STOP"

                fnp = getattr(self.mt5, "pending_order", None) or getattr(self.mt5, "enviar_ordem_pendente", None)
                if callable(fnp):
                    try:
                        return fnp(kind, price=entry, sl=sl, tp=tp, lots=lots)
                    except TypeError:
                        return fnp(order_kind=kind, price=entry, sl=sl, tp=tp, lots=lots)

                print(f"[SIM] {kind} price={entry} sl={sl} tp={tp} lots={lots}")
                return {"status": "simulado", "type": kind, "price": entry, "sl": sl, "tp": tp, "lots": lots}
            except Exception as e:
                print("[AI order wrapper] erro:", e)
                return {"status": "erro", "err": str(e)}
        self._ai_send_order_wrapper = _ai_send_order_wrapper
        # ----------------------------------------------------------------

        # Copiloto AI (chat/teach/order textual – opcional)
        self.AI = TraderAI(
            trader=self.mt5,
            send_order_fn=self._ai_send_order_wrapper,
            sanitize_fn=None,
            symbol=self.vt.cfg.symbol
        )

        # >>> Controlador de risco/fluxo (martingale + 1 posição) — SEM cooldown interno
        self.trade_ctl = TradeControl(TradeControlConfig(
            symbol=self.vt.cfg.symbol,
            magic=self.vt.cfg.magic,
            base_lot=self.vt.cfg.lot,
            martingale_factor=3.0,
            max_lot=99.50,
            history_days=1,
            one_position_only=True,
        ))

        # =====================[ COOL­DOWN PÓS-LOSS (orquestrado no main) ]=====================
        self.cooldown_until = None       # horário (local) até quando aguarda
        self.cooldown_armed = False      # evita rearmar cooldown para o MESMO loss
        # =====================================================================================

    def dentro_do_horario(self) -> bool:
        """
        Janela 24x5 de Forex em UTC:
          - Abre:  domingo 22:00 UTC
          - Fecha: sexta   22:00 UTC
        Mon=0 .. Sun=6 (datetime.utcnow().weekday()).
        """
        now = datetime.utcnow()
        w = now.weekday()  # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
        hm = now.hour + now.minute / 60.0

        if w in (0, 1, 2, 3):  # segunda a quinta
            return True
        if w == 4:             # sexta até 22:00 UTC
            return hm < 22.0
        if w == 6:             # domingo a partir de 22:00 UTC
            return hm >= 22.0
        return False           # sábado

    def _sleep_until_next_5min(self) -> None:
        """Espera até :00, :05, :10, ... do relógio do sistema."""
        now = datetime.now()
        add_min = (5 - (now.minute % 5)) % 5
        if add_min == 0 and now.second == 0 and now.microsecond == 0:
            return
        if add_min == 0:
            add_min = 5
        nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=add_min)
        time.sleep((nxt - now).total_seconds())

    def _unique_by_timeframe(self, usados):
        """
        (DESABILITADO) Antes filtrava 1 item por timeframe.
        Agora apenas repassa a lista original e marca como 'já era único'.
        """
        return list(usados), True

    def _log_sinal(self, r: dict, params: dict, entry: float, stop: float, target: float, atr: float) -> None:
        """Registra sinal detectado com status PENDING para avaliação futura."""
        registro = {
            "detect_time": datetime.utcnow(),
            "timeframe":   r["Timeframe"],
            "pattern":     r["Padrão"],
            "signal":      r["Sinal"],
            "prob":        r["Probabilidade"],
            "rsi":         r["RSI"],
            "ema":         r["EMA"],
            "vwap":        r["VWAP"],
            "price":       r["Preço"],
            "volume":      r["Volume"],
            "entry":       r["Entry"],
            "stop":        r["Stop"],
            "target":      r["Target"],
            "atr":         atr,
            "rr":          params["RR_MULT"],
            "status":      "PENDING",
            "evaluated_at": None,
            "params":      json.dumps(params),
        }
        self.tuner.salvar_sinal(registro)

    def rodar(self) -> None:
        """Executa o loop contínuo do scanner, com auto-tuning e veredito final."""
        self.mt5.conectar()

        while True:
            # 0) Alinha ao relógio do sistema (próximo :00/:05/:10/…)
            self._sleep_until_next_5min()

            # 0.1) Janela operacional — se fechado, apenas pula para o próximo batimento
            if not self.dentro_do_horario():
                print("⏱️ Fora do horário operacional.")
                continue

            # =====================[ GATE DO COOL­DOWN ]====================
            if self.cooldown_until:
                now = datetime.now()
                if now < self.cooldown_until:
                    restantes = int((self.cooldown_until - now).total_seconds() // 60)
                    print(f"❄️ Em cooldown até {self.cooldown_until.strftime('%H:%M:%S')} (faltam {restantes} min).")
                    continue  # << NÃO sai do loop: apenas aguarda o próximo batimento
                else:
                    print("✅ Cooldown finalizado — retomando operações automáticas.")
                    self.cooldown_until = None
                    # Mantemos cooldown_armed=True até uma nova operação ser executada com sucesso
            # =============================================================

            # 1) Avaliar sinais pendentes e recalibrar parâmetros
            atualizados = self.tuner.avaliar_pendentes(self.mt5)
            if atualizados:
                print(f"📝 Avaliados {atualizados} sinais pendentes.")
            params = self.tuner.ajustar_parametros()

            print(f"\n🔎 {datetime.utcnow()} → Scanner Multi-Timeframe | Params: {params}")
            resultados = []

            # 2) Varrida por timeframe com múltiplos padrões
            for nome, tf in TIMEFRAMES.items():
                df = self.mt5.obter_candles(tf, QTD_CANDLES)
                if df.empty:
                    continue

                df = Indicators.aplicar(df, ema_period=params["EMA_PERIOD"])

                padroes = PatternDetector.detect_all(
                    df, tolerancia=params["TOLERANCIA"], params=DETECTOR_PARAMS
                )
                if not padroes:
                    continue

                for padrao in padroes:
                    prob, sinal = Analyzer.calcular_probabilidade_por_padrao(
                        df, padrao,
                        rsi_buy=params["RSI_SOBREVENDIDO"],
                        rsi_sell=params["RSI_SOBRECOMPRA"]
                    )
                    entry, stop, target, atr = Analyzer.montar_trade_levels(
                        df, sinal, atr_mult=params["ATR_MULT_STOP"], rr=params["RR_MULT"]
                    )
                    r = {
                        "Timeframe": nome, "Padrão": padrao, "Sinal": sinal, "Probabilidade": prob,
                        "RSI": round(df["RSI"].iloc[-1], 2), "EMA": round(df["EMA"].iloc[-1], 2),
                        "VWAP": round(df["VWAP"].iloc[-1], 2), "Preço": round(df["close"].iloc[-1], 2),
                        "Volume": int(df["tick_volume"].iloc[-1]),
                        "Entry": round(entry, 2), "Stop": round(stop, 2), "Target": round(target, 2),
                    }
                    resultados.append(r)
                    self.historico_padroes.append(padrao)
                    print(f"{r['Timeframe']} | {r['Padrão']} | {r['Sinal']} "
                          f"| Prob: {r['Probabilidade']}% | Entry:{r['Entry']} "
                          f"| Stop:{r['Stop']} | Target:{r['Target']}")
                    print(f"Indicadores → RSI:{r['RSI']} | EMA:{r['EMA']} | VWAP:{r['VWAP']} "
                          f"| Preço:{r['Preço']} | Vol:{r['Volume']}")
                    self._log_sinal(r, params, entry, stop, target, atr)

            # 3) Consolida e decide
            if resultados:
                ranking = pd.Series(self.historico_padroes).value_counts().to_dict()
                print(f"\n📊 Ranking (sessão): {ranking}")

                veredito_local, score, usados = VerdictAggregator.decidir(
                    resultados, min_prob=MIN_PROB_CONSENSO, pesos=PESOS_TIMEFRAMES
                )
                print(f"🧭 Veredito Local (min_prob={MIN_PROB_CONSENSO}): {veredito_local} | score={score:.1f}")

                usados_unique, ja_era_unico = self._unique_by_timeframe(usados)
                if usados_unique:
                    for tf, pad, sig, p, w in usados_unique:
                        print(f" - {tf}: {pad} | {sig} | {int(round(p))}% (peso {w})")

                self.notifier.evaluate_and_notify(
                    veredito_local=veredito_local,
                    score=score,
                    usados=usados_unique,
                    min_prob_cfg=MIN_PROB_CONSENSO,
                )

                # >>> EXECUÇÃO DE ORDEM PELO SCORE
                if ja_era_unico:
                    # --- SINCRONIZA ESTADO DO CONTROLE (histórico e posições)
                    self.trade_ctl.sync()

                    # Regra: 1 posição por vez
                    if self.trade_ctl.is_position_open():
                        print("⏸️ Já existe posição aberta — aguardando encerramento para nova entrada.")
                    else:
                        old_lot = float(self.vt.cfg.lot)
                        try:
                            next_lot = float(self.trade_ctl.get_next_lot())
                        except Exception:
                            next_lot = old_lot

                        # === DISPARO DO COOL­DOWN: detecta LOSS (next_lot > base) e arma um único cooldown ===
                        if (next_lot > old_lot) and (not self.cooldown_armed):
                            self.cooldown_until = datetime.now() + timedelta(minutes=6)
                            self.cooldown_armed = True
                            print(f"⏸️ Loss detectado. Cooldown de 6 minutos até "
                                  f"{self.cooldown_until.strftime('%H:%M:%S')} (hora do sistema).")
                            continue  # volta ao topo; ao expirar, o gate libera

                        # Caso contrário, opera normalmente
                        self.vt.cfg.lot = next_lot
                        trade = self.vt.decide_and_execute(score)
                        self.vt.cfg.lot = old_lot

                        if trade.get("ok"):
                            print(f"✅ Trade executado: {trade}")
                            # Libera o próximo cooldown apenas após uma nova operação bem-sucedida
                            self.cooldown_armed = False
                        else:
                            print(f"ℹ️  Sem execução: {trade.get('reason')}")
                else:
                    print("⛔ Execução automática bloqueada: veredito com TFs repetidos.")

                # (opcional) trailing stop nas posições abertas (se trail_points > 0)
                self.vt.manage_trailing()

                # Gate opcional por força mínima do consenso (apenas log)
                if abs(score) < MIN_SCORE_TRADE:
                    print(f"⚠️ Score insuficiente ({score:.1f} < {MIN_SCORE_TRADE}). Aguardar confirmação.")

                # Resumo para IA (DeepSeek)
                resumo = "Análise técnica XAU/USD:\n"
                for r in resultados:
                    resumo += (f"{r['Timeframe']} → {r['Padrão']} | {r['Sinal']} | Prob:{r['Probabilidade']}% | "
                               f"RSI:{r['RSI']} | EMA:{r['EMA']} | VWAP:{r['VWAP']} | Preço:{r['Preço']} | "
                               f"Entry:{r['Entry']} | Stop:{r['Stop']} | Target:{r['Target']}\n")
                resumo += f"Parâmetros atuais: {params}\nPadrões mais frequentes: {ranking}\n"
                resumo += f"Veredito Local: {veredito_local} (score={score:.1f})"

                veredito = self.deepseek.analisar(resumo, retries=2, timeout=30, backoff=4)
                if isinstance(veredito, str) and veredito.startswith("Erro DeepSeek"):
                    print(f"\n🤖 DeepSeek ERRO → {veredito}")
                    print(f"Fallback (veredito local) → {veredito_local}")
                else:
                    print(f"\n🤖 DeepSeek Veredito Final: {veredito}")
            else:
                print("Nenhum padrão detectado.")

            # 👉 o alinhamento é feito no topo do loop; não dormir aqui


if __name__ == "__main__":
    M = Monitor()
    M.rodar()
