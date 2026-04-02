# FILE NAME: main.py
"""main
Ponto de entrada: orquestra o scanner multi‑timeframe, auto‑tuning, consenso local
e veredito do DeepSeek (com fallback). Agora suporta múltiplos padrões clássicos.

Execute:
    python main.py
"""

import logging
import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from Sincronizador_de_horario import sync_windows_time

from config import (
    DEEPSEEK_API_KEY,
    INTERVALO_MINUTOS,
    TIMEFRAMES,
    QTD_CANDLES,
    MIN_PROB_CONSENSO,
    PESOS_TIMEFRAMES,
    MIN_SCORE_TRADE,
    DETECTOR_PARAMS,
    TOLERANCIA_PADRAO,
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

from core.verdict_trader import VerdictTrader, VerdictTraderConfig
from core.trade_control import TradeControl, TradeControlConfig

# --- Credenciais do Telegram (via ENV, com fallback) ---
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "8288271011:AAHN85gL63VB6t1z4ER8jdbb-19_lGK6hSM") # Seu token do bot
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003877903665")

os.environ.setdefault("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
os.environ.setdefault("TELEGRAM_TOKEN", TELEGRAM_TOKEN)
os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)

print("🔧 Padrões buscados:", ", ".join(PatternDetector.listar_padroes()))

# Configura o logger da aplicação
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("Megazord")

class Monitor:
    """Loop principal: coleta dados, detecta padrões, calcula prob/sinal,
    loga e decide veredito."""

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

        # Execução automática pelo score (BUY/SELL)
        self.vt = VerdictTrader(VerdictTraderConfig())
        ok_conn = self.vt.connect()
        if ok_conn:
            print("[VerdictTrader] ✅ Conectado ao MT5 e símbolo preparado.")
        else:
            print("[VerdictTrader] ❌ Falha de conexão/login no MT5.")

        # Adaptador de envio de ordens para o TraderAI
        def _ai_send_order_wrapper(trader, side, px_now, entry, sl, tp, lots=None, near_pct=0.0001):
            try:
                lots = lots or 0.02
                if max(px_now, 1) > 0 and entry is not None and abs(entry - px_now) / px_now < float(near_pct):
                    fn = getattr(self.mt5, "market_order", None) or getattr(self.mt5, "enviar_ordem_market", None)
                    if callable(fn):
                        return fn(side=side, lots=lots, sl=sl, tp=tp)
                    print(f"[SIM] MARKET {side} lots={lots} sl={sl} tp={tp}")
                    return {"status": "simulado", "type": "MARKET", "side": side, "lots": lots, "sl": sl, "tp": tp}

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

        # Copiloto AI (opcional)
        self.AI = TraderAI(
            trader=self.mt5,
            send_order_fn=self._ai_send_order_wrapper,
            sanitize_fn=None,
            symbol=self.vt.cfg.symbol
        )

        # Controlador de risco/fluxo
        self.trade_ctl = TradeControl(TradeControlConfig(
            symbol=self.vt.cfg.symbol,
            magic=self.vt.cfg.magic,
            base_lot=self.vt.cfg.lot,
            martingale_factor=1.0,
            max_lot=99.50,
            history_days=1,
            one_position_only=False,
        ))

    def dentro_do_horario(self) -> bool:
        """Verifica se está dentro do horário operacional."""
        now = datetime.utcnow()
        w = now.weekday()
        hm = now.hour + now.minute / 60.0

        if w in (0, 1, 2, 3):
            return True
        if w == 4:
            return hm < 22.0
        if w == 6:
            return hm >= 22.0
        return False

    def _sleep_until_next_5min(self) -> None:
        """Espera até o próximo batimento de 5 minutos do relógio local."""
        now = datetime.now()
        add_min = (15 - (now.minute % 15)) % 15
        if add_min == 0 and now.second == 0 and now.microsecond == 0:
            return
        if add_min == 0:
            add_min = 15
        nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=add_min)
        time.sleep((nxt - now).total_seconds())

    def _unique_by_timeframe(self, usados):
        """Filtra sinais únicos por timeframe."""
        filtrado = []
        ja_era_unico = True
        for item in usados:
            if item not in filtrado:
                filtrado.append(item)
            else:
                ja_era_unico = False
        return filtrado, ja_era_unico

    def _log_sinal(self, r: dict, params: dict, entry: float, stop: float, target: float, atr: float) -> None:
        """Registra sinal detectado com status PENDING para avaliação futura."""
        registro = {
            "detect_time": datetime.utcnow(),
            "timeframe": r["Timeframe"],
            "pattern": r["Padrão"],
            "signal": r["Sinal"],
            "prob": r["Probabilidade"],
            "rsi": r["RSI"],
            "ema": r["EMA"],
            "vwap": r["VWAP"],
            "price": r["Preço"],
            "volume": r["Volume"],
            "entry": r["Entry"],
            "stop": r["Stop"],
            "target": r["Target"],
            "atr": atr,
            "rr": params["RR_MULT"],
            "status": "PENDING",
            "evaluated_at": None,
            "params": json.dumps(params),
        }
        self.tuner.salvar_sinal(registro)

    def rodar(self) -> None:
        """Executa o loop contínuo do scanner, com auto-tuning e veredito final."""
        self.mt5.conectar()

        while True:
            sync_windows_time() # sincroniza o horario
            self._sleep_until_next_5min()

            if not self.dentro_do_horario():
                print("⏱️ Fora do horário operacional.")
                continue

            atualizados = self.tuner.avaliar_pendentes(self.mt5)
            if atualizados:
                print(f"📝 Avaliados {atualizados} sinais pendentes.")
            params = self.tuner.ajustar_parametros()

            print(f"\n🔎 {datetime.utcnow()} → Scanner Multi-Timeframe | Params: {params}")
            resultados = []

            for nome, tf in TIMEFRAMES.items():
                df = self.mt5.obter_candles(tf, QTD_CANDLES)
                if df.empty:
                    continue

                df = Indicators.aplicar(df, ema_period=params["EMA_PERIOD"])
                padroes = PatternDetector.detect_all(df, tolerancia=params["TOLERANCIA"], params=DETECTOR_PARAMS)
                if not padroes:
                    continue

                for padrao in padroes:
                    prob, sinal = Analyzer.calcular_probabilidade_por_padrao(df, padrao,
                        rsi_buy=params["RSI_SOBREVENDIDO"],
                        rsi_sell=params["RSI_SOBRECOMPRA"],
                    )

                    entry, stop, target, atr = Analyzer.montar_trade_levels(df, sinal,
                        atr_mult=params["ATR_MULT_STOP"], rr=params["RR_MULT"]
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

                    print(f"{r['Timeframe']} | {r['Padrão']} | {r['Sinal']} | Prob: {r['Probabilidade']}% | Entry:{r['Entry']} | Stop:{r['Stop']} | Target:{r['Target']}")
                    print(f"Indicadores → RSI:{r['RSI']} | EMA:{r['EMA']} | VWAP:{r['VWAP']} | Preço:{r['Preço']} | Vol:{r['Volume']}")

                    self._log_sinal(r, params, entry, stop, target, atr)

            if resultados:
                ranking = pd.Series(self.historico_padroes).value_counts().to_dict()
                print(f"\n📊 Ranking (sessão): {ranking}")

                veredito_local, score, usados = VerdictAggregator.decidir(resultados,
                    min_prob=MIN_PROB_CONSENSO, pesos=PESOS_TIMEFRAMES
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

                # --- BLOQUEIO PARA NÃO CONSULTAR DEEPSEEK SE JÁ HOUVER POSIÇÃO ABERTA ---
                self.trade_ctl.sync()  # sincroniza posições abertas no MT5
                if self.trade_ctl.is_position_open():
                    print("⏸️ Já existe posição aberta — DeepSeek NÃO será consultado neste ciclo.")
                    # pula todo o restante (não monta resumo, não chama DeepSeek, não envia ordem)
                    continue
                # -------------------------------------------------------------------------

                # A PARTIR DAQUI: só executa se NÃO houver posição aberta
                old_lot = float(self.vt.cfg.lot)
                try:
                    self.vt.cfg.lot = float(self.trade_ctl.get_next_lot())
                except Exception:
                    self.vt.cfg.lot = old_lot

                resumo = "Análise técnica XAU/USD:\n"
                for r in resultados:
                    resumo += (
                        f"{r['Timeframe']} → {r['Padrão']} | {r['Sinal']} | "
                        f"Prob:{r['Probabilidade']}% | RSI:{r['RSI']} | EMA:{r['EMA']} | "
                        f"VWAP:{r['VWAP']} | Preço:{r['Preço']} | Entry:{r['Entry']} | "
                        f"Stop:{r['Stop']} | Target:{r['Target']}\n"
                    )
                resumo += f"Parâmetros atuais: {params}\nPadrões mais frequentes: {ranking}\n"

                # 👉 Consulta à API do DeepSeek SÓ ACONTECE se chegou até aqui (ou seja, sem posição aberta)
                deepseek_response = self.deepseek.analisar_trade(resumo)

                if "error" in deepseek_response:
                    print(f"\n🤖 DeepSeek ERRO → {deepseek_response['error']}")
                    print(f"Fallback (veredito local) → {veredito_local}")

                    side_to_trade = None
                    if "COMPRA" in veredito_local.upper():
                        side_to_trade = "BUY"
                    elif "VENDA" in veredito_local.upper():
                        side_to_trade = "SELL"

                    # no fallback você pode depois trocar SL/TP por algo calculado localmente
                    sl = 0.0
                    tp = 0.0
                    logger.info(f"Motivo da decisão (Fallback): {veredito_local}")
                else:
                    print(f"\n🤖 DeepSeek Veredito Final: {deepseek_response['veredito_text']}")
                    side_to_trade = None
                    if deepseek_response['decision'] == "COMPRA":
                        side_to_trade = "BUY"
                    elif deepseek_response['decision'] == "VENDA":
                        side_to_trade = "SELL"
                    sl = deepseek_response['sl']
                    tp = deepseek_response['tp']
                    logger.info(f"Motivo da decisão do DeepSeek: {deepseek_response['reason']}")

                if side_to_trade:
                    # Checagem FINAL de segurança (caso algo tenha aberto posição entre a análise e o envio)
                    if self.trade_ctl.is_position_open():
                        print("⏸️ Já existe posição aberta — decisão DeepSeek ignorada (bloqueio final).")
                    else:
                        # no seu código original o método ainda é _get_current_price
                        price_now = self.vt._get_current_price(side_to_trade)
                        if price_now is None:
                            print("⚠️ Não foi possível obter preço atual – ordem não enviada.")
                        else:
                            order_res = self.vt._order_send_market(side_to_trade, price_now, sl, tp)
                            if order_res and order_res.get("ok"):
                                print(f"✅ Ordem {side_to_trade} enviada por decisão DeepSeek:")
                                print(f"    preço  = {price_now:.5f}")
                                print(f"    SL     = {sl:.5f}")
                                print(f"    TP     = {tp:.5f}")
                                print(
                                    "    ticket = "
                                    f"{order_res.get('result').order if order_res.get('result') else 'N/A'}"
                                )
                            else:
                                print(f"❌ Falha ao enviar ordem {side_to_trade} por decisão DeepSeek:")
                                print(f"    detalhe = {order_res}")
                else:
                    print(f"\n🤖 DeepSeek Veredito Final (sem ação): {deepseek_response['veredito_text']}")
            else:
                print("Nenhum padrão detectado.")


if __name__ == "__main__":
    M = Monitor()

    # -------------------------------------------------
    #  NÃO CHAMA O TRADERAI – evita a mensagem “Não há pergunta clara”
    # -------------------------------------------------
    # resp = M.AI.chat("Qual é a resposta ? ?.")
    # print("🤖 Qual a sua atuação aqui ?:", resp.get("reply") if isinstance(resp, dict) else resp)

    M.rodar()

