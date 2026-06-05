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
import sys # Importar sys

from config import (
    OLLAMA_API_BASE_URL,
    OLLAMA_MODEL_NAME,
    INTERVALO_MINUTOS,
    TIMEFRAMES,
    QTD_CANDLES,
    MIN_PROB_CONSENSO,
    PESOS_TIMEFRAMES,
    MIN_SCORE_TRADE,
    DETECTOR_PARAMS,
    TOLERANCIA_PADRAO,
    VERDICT_FINAL_SOURCE,
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

# Export Ollama settings to env (optional) so other libs can read them
os.environ.setdefault("OLLAMA_API_BASE_URL", str(OLLAMA_API_BASE_URL))
os.environ.setdefault("OLLAMA_MODEL_NAME", str(OLLAMA_MODEL_NAME))
os.environ.setdefault("TELEGRAM_TOKEN", TELEGRAM_TOKEN)
os.environ.setdefault("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)

print("Padroes buscados:", ", ".join(PatternDetector.listar_padroes()))

# Configura o logger da aplicação para Python 3.8
logger = logging.getLogger("Megazord")
logger.setLevel(logging.INFO)

# Remove handlers existentes para evitar duplicação se o script for executado múltiplas vezes
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)

# Cria um StreamHandler que escreve para sys.stdout
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)


def _print_deepseek_report(run_ts, params, resultados, ranking, veredito_local, deepseek_response, pending_count=None):
    """Pretty-print consolidated scanner + DeepSeek report to terminal."""
    sep = "--------------------------------------------------------------"
    print(sep)
    print(f"SCANNER {run_ts} - Multi-Timeframe")
    print("Params:", params)
    print(sep)

    # Timeframes already printed earlier in the loop; print ranking and local verdict compactly
    ranking_str = " | ".join(f"{k}: {v}" for k, v in ranking.items()) if ranking else "-"
    print(f"Ranking (sessao): {ranking_str}")
    print(f"Veredito Local (min_prob={veredito_local.get('min_prob', '?')}): {veredito_local.get('decision','?')} | score={veredito_local.get('score', 0.0)}")

    # DeepSeek response summary
    print()
    if not deepseek_response:
        print("DeepSeek Veredito Final: <nenhuma resposta>")
        return

    # If we attached canonical final JSON, use it
    final = deepseek_response.get('final') if isinstance(deepseek_response, dict) else None
    if final and isinstance(final, dict):
        print("LOGS (IA)")
        # request/response timing lines are already logged elsewhere; show final compact
        print(f"DeepSeek - Veredito Final (compact): {json.dumps(final, ensure_ascii=False)}")
        # also print human-friendly
        print(f"- DECISAO: {final.get('decision')}  |  SL: {final.get('sl')}  |  TP: {final.get('tp')}  | source: {final.get('source')}")
    else:
        # fallback: try to parse veredito_text
        vt = deepseek_response.get('veredito_text') if isinstance(deepseek_response, dict) else str(deepseek_response)
        print("DeepSeek Veredito Final:")
        print(vt)

    if pending_count is not None:
        print()
        print(f"Avaliados {pending_count} sinais pendentes.")
    print(sep)


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
            logger.info("Conectado ao MT5 e simbolo preparado.")
        else:
            logger.error("Falha de conexao/login no MT5.")

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
        add_min = (5 - (now.minute % 5)) % 5
        if add_min == 0 and now.second == 0 and now.microsecond == 0:
            return
        if add_min == 0:
            add_min = 5
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
                logger.info("Fora do horario operacional.")
                continue

            atualizados = self.tuner.avaliar_pendentes(self.mt5)
            if atualizados:
                logger.info(f"Avaliados {atualizados} sinais pendentes.")
            params = self.tuner.ajustar_parametros()

            logger.info(f"\nScanner Multi-Timeframe | Params: {params}")
            resultados = []
            processed_dfs = {}

            for nome, tf in TIMEFRAMES.items():
                df = self.mt5.obter_candles(tf, QTD_CANDLES)
                if df.empty:
                    continue

                df = Indicators.aplicar(df, ema_period=params["EMA_PERIOD"])
                # armazenar df processado para checagens posteriores (ex: flat MA / proximidade)
                processed_dfs[nome] = df
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

                    logger.info(f"{r['Timeframe']} | {r['Padrão']} | {r['Sinal']} | Prob: {r['Probabilidade']}% | Entry:{r['Entry']} | Stop:{r['Stop']} | Target:{r['Target']}")
                    logger.info(f"Indicadores -> RSI:{r['RSI']} | EMA:{r['EMA']} | VWAP:{r['VWAP']} | Preco:{r['Preço']} | Vol:{r['Volume']}")

                    self._log_sinal(r, params, entry, stop, target, atr)

            if resultados:
                ranking = pd.Series(self.historico_padroes).value_counts().to_dict()
                logger.info(f"\nRanking (sessao): {ranking}")

                veredito_local, score, usados = VerdictAggregator.decidir(resultados,
                    min_prob=MIN_PROB_CONSENSO, pesos=PESOS_TIMEFRAMES
                )
                logger.info(f"Veredito Local (min_prob={MIN_PROB_CONSENSO}): {veredito_local} | score={score:.1f}")

                usados_unique, ja_era_unico = self._unique_by_timeframe(usados)

                if usados_unique:
                    for tf, pad, sig, p, w in usados_unique:
                        logger.info(f" - {tf}: {pad} | {sig} | {int(round(p))}% (peso {w})")

                self.notifier.evaluate_and_notify(
                    veredito_local=veredito_local,
                    score=score,
                    usados=usados_unique,
                    min_prob_cfg=MIN_PROB_CONSENSO,
                )

                # --- BLOQUEIO PARA NAO CONSULTAR DEEPSEEK SE JA HOUVER POSICAO ABERTA ---
                self.trade_ctl.sync()  # sincroniza posições abertas no MT5
                if self.trade_ctl.is_position_open():
                    logger.info("Ja existe posicao aberta - DeepSeek NAO sera consultado neste ciclo.")
                    # pula todo o restante (nao monta resumo, nao chama DeepSeek, nao envia ordem)
                    continue
                # -------------------------------------------------------------------------

                # A PARTIR DAQUI: so executa se NAO houver posicao aberta
                old_lot = float(self.vt.cfg.lot)
                try:
                    self.vt.cfg.lot = float(self.trade_ctl.get_next_lot())
                except Exception:
                    self.vt.cfg.lot = old_lot

                # Build a compact resumo: send only the 2 best signals to the model to reduce context/token use.
                # Prefer M15 and H1 if present; otherwise pick top-2 by probability. Also compute mean prob to decide
                # whether to force a decision (no AGUARDAR).
                resumo = "Analise tecnica XAU/USD:\n"
                # find candidates
                by_tf = {r['Timeframe']: r for r in resultados}
                chosen = []
                # prefer M15 and H1
                for pref in ("M15", "H1"):
                    if pref in by_tf:
                        chosen.append(by_tf[pref])
                # fill up to 2 with highest prob
                if len(chosen) < 2:
                    remaining = [r for r in resultados if r not in chosen]
                    remaining_sorted = sorted(remaining, key=lambda x: x['Probabilidade'], reverse=True)
                    for r in remaining_sorted:
                        if len(chosen) >= 2:
                            break
                        chosen.append(r)

                # If none selected (edge case), fallback to first two
                if not chosen:
                    chosen = resultados[:2]

                for r in chosen:
                    resumo += (
                        f"{r['Timeframe']} -> {r['Padrão']} | {r['Sinal']} | "
                        f"Prob:{r['Probabilidade']}% | RSI:{r['RSI']} | EMA:{r['EMA']} | "
                        f"VWAP:{r['VWAP']} | Preco:{r['Preço']} | Entry:{r['Entry']} | "
                        f"Stop:{r['Stop']} | Target:{r['Target']}\n"
                    )
                # compute mean probability for chosen signals
                try:
                    mean_prob = sum(float(r['Probabilidade']) for r in chosen) / len(chosen)
                except Exception:
                    mean_prob = 0.0
                resumo += f"Parametros atuais: {params}\nPadroes mais frequentes: {ranking}\n"

                # --- FILTRO MULTI-TF: verifica EMA flat antes de gastar tokens/CPU ---
                try:
                    from config import VERDICT_FLAT_FILTER_ENABLE, VERDICT_FLAT_TFS, VERDICT_FLAT_MODE, VERDICT_FLAT_LOOKBACK
                    if VERDICT_FLAT_FILTER_ENABLE:
                        any_flat = False
                        for tf_key in VERDICT_FLAT_TFS:
                            df_for_tf = processed_dfs.get(tf_key)
                            if self.vt.is_market_flat(df_for_tf, mode=VERDICT_FLAT_MODE, lookback=VERDICT_FLAT_LOOKBACK):
                                msg = f"Ciclo abortado: EMA esta plana em {tf_key} (modo={VERDICT_FLAT_MODE}) - DeepSeek NAO sera consultado."
                                logger.info(msg)
                                try:
                                    log_path = os.path.join(os.getcwd(), "Logs", "log_api.log")
                                    self.deepseek._rotate_and_append(log_path, f"PRECHECK_ABORT: {msg}\n")
                                except Exception:
                                    pass
                                any_flat = True
                                break
                        if any_flat:
                            logger.info("Ciclo atual abortado devido a filtro de EMA plana. Nenhuma decisao ou ordem sera processada.")
                            continue
                except Exception:
                    logger.debug("Falha na checagem EMA flat antes do DeepSeek")

                # --- FILTRO DE PROXIMIDADE MA9: Pre-IA (economiza tokens) ---
                try:
                    from config import VERDICT_PROXIMITY_FILTER_ENABLE
                    if VERDICT_PROXIMITY_FILTER_ENABLE:
                        df_m15 = processed_dfs.get("M15")
                        if df_m15 is not None and len(df_m15) > 0:
                            preco_atual = float(df_m15["close"].iloc[-1])
                            if not self.vt.is_price_within_proximity(df_m15, preco_atual):
                                msg = (f"[FILTRO PROXIMIDADE] Ciclo Pre-IA abortado em M15: "
                                       f"Preco atual ({preco_atual:.2f}) distante da EMA9. DeepSeek poupado.")
                                logger.info(msg)
                                try:
                                    log_path = os.path.join(os.getcwd(), "Logs", "log_api.log")
                                    self.deepseek._rotate_and_append(log_path, f"PRECHECK_ABORT: {msg}\n")
                                except Exception:
                                    pass
                                logger.info("Ciclo atual abortado devido a filtro de proximidade. Nenhuma decisao ou ordem sera processada.")
                                continue
                except Exception:
                    logger.debug("Falha na checagem de proximidade pre-IA")

                # Decisao final: pode ser 'local', 'deepseek' ou 'hybrid' (configurado em config.VERDICT_FINAL_SOURCE)
                side_to_trade = None
                sl = 0.0
                tp = 0.0

                if VERDICT_FINAL_SOURCE == "local":
                    # Nao chama a IA - segue veredito local
                    logger.info("VERDICT_FINAL_SOURCE=local -> seguindo veredito local sem consultar DeepSeek.")
                    if "COMPRA" in veredito_local.upper():
                        side_to_trade = "BUY"
                    elif "VENDA" in veredito_local.upper():
                        side_to_trade = "SELL"

                    if side_to_trade and resultados:
                        # escolhe um candidato representativo para SL/TP (primeiro com mesmo sinal ou primeiro geral)
                        candidate = None
                        for cand in resultados:
                            try:
                                if cand.get("Sinal") and cand["Sinal"].upper() in veredito_local.upper():
                                    candidate = cand
                                    break
                            except Exception:
                                continue
                        if candidate is None:
                            candidate = resultados[0]
                        sl = candidate.get("Stop", 0.0)
                        tp = candidate.get("Target", 0.0)

                else:
                    # Chama DeepSeek (modo 'deepseek' ou 'hybrid')
                    try:
                        log_path = os.path.join(os.getcwd(), "Logs", "log_api.log")
                        self.deepseek._rotate_and_append(log_path, f"CALLING_DEEPSEEK: resumo_len={len(resumo)}\n")
                    except Exception:
                        pass
                    # Use a low timeout for regular cycles to avoid long stalls; retry logic exists in the client
                    # If mean_prob > 70, force the model to choose COMPRA or VENDA (no AGUARDAR)
                    decision_required = True if mean_prob > 70.0 else False
                    deepseek_response = self.deepseek.analisar_trade(resumo, timeout=4.5, decision_required=decision_required)

                    if "error" in deepseek_response:
                        logger.error(f"DeepSeek ERRO -> {deepseek_response['error']}")
                        if VERDICT_FINAL_SOURCE == "hybrid":
                            # fallback para veredito local (comportamento legacy)
                            logger.info(f"Fallback (veredito local) -> {veredito_local}")
                            if "COMPRA" in veredito_local.upper():
                                side_to_trade = "BUY"
                            elif "VENDA" in veredito_local.upper():
                                side_to_trade = "SELL"
                            if side_to_trade and resultados:
                                # escolhe um candidato representativo para SL/TP
                                candidate = None
                                for cand in resultados:
                                    try:
                                        if cand.get("Sinal") and cand["Sinal"].upper() in veredito_local.upper():
                                            candidate = cand
                                            break
                                    except Exception:
                                        continue
                                if candidate is None:
                                    candidate = resultados[0]
                                sl = candidate.get("Stop", 0.0)
                                tp = candidate.get("Target", 0.0)
                            logger.info(f"Motivo da decisao (Fallback): {veredito_local}")
                        else:
                            # modo 'deepseek' exige resposta da IA - sem fallback
                            logger.warning("VERDICT_FINAL_SOURCE=deepseek -> sem fallback. Nenhuma acao sera tomada.")
                            side_to_trade = None

                    else:
                        # Pretty-print consolidated report including DeepSeek final JSON (if available)
                        try:
                            _print_deepseek_report(datetime.utcnow(), params, resultados, ranking if 'ranking' in locals() else {}, {"min_prob": MIN_PROB_CONSENSO, "decision": veredito_local, "score": score}, deepseek_response, pending_count=atualizados if atualizados else None)
                        except Exception:
                            # fallback to previous behavior
                            logger.info(f"DeepSeek Veredito Final: {deepseek_response.get('veredito_text', '<sem texto>')}")
                        if deepseek_response.get('decision') == "COMPRA":
                            side_to_trade = "BUY"
                        elif deepseek_response.get('decision') == "VENDA":
                            side_to_trade = "SELL"
                        sl = deepseek_response.get('sl', 0.0)
                        tp = deepseek_response.get('tp', 0.0)
                        logger.info("Motivo da decisao do DeepSeek: %s", deepseek_response.get('reason'))

                if side_to_trade:
                    # Checagem FINAL de seguranca (caso algo tenha aberto posicao entre a analise e o envio)
                    if self.trade_ctl.is_position_open():
                        logger.info("Ja existe posicao aberta - decisao DeepSeek ignorada (bloqueio final).")
                    else:
                        # no seu codigo original o metodo ainda e _get_current_price
                        price_now = self.vt._get_current_price(side_to_trade)
                        if price_now is None:
                            logger.warning("Nao foi possivel obter preco atual - ordem nao enviada.")
                        else:
                            order_res = self.vt._order_send_market(side_to_trade, price_now, sl, tp)
                            if order_res and order_res.get("ok"):
                                logger.info(f"Ordem {side_to_trade} enviada por decisao {'DeepSeek' if VERDICT_FINAL_SOURCE != 'local' else 'Local'}:")
                                logger.info(f"    preco  = {price_now:.5f}")
                                logger.info(f"    SL     = {sl:.5f}")
                                logger.info(f"    TP     = {tp:.5f}")
                                logger.info(
                                    "    ticket = "
                                    f"{order_res.get('result').order if order_res.get('result') else 'N/A'}"
                                )
                            else:
                                logger.error(f"Falha ao enviar ordem {side_to_trade} por decisao {'DeepSeek' if VERDICT_FINAL_SOURCE != 'local' else 'Local'}:")
                                logger.error(f"    detalhe = {order_res}")
                else:
                    # Evita referenciar `deepseek_response` quando nao foi definido (modo 'local')
                    if VERDICT_FINAL_SOURCE == "local":
                        logger.info(f"\nVeredito final (local) sem acao: {veredito_local}")
                    else:
                        # Se estiver no modo 'deepseek' ou 'hybrid', tente mostrar a resposta da IA se disponivel
                        if 'deepseek_response' in locals() and isinstance(deepseek_response, dict):
                            logger.info(f"\nDeepSeek Veredito Final (sem acao): {deepseek_response.get('veredito_text', '<sem texto>')}")
                        else:
                            logger.info("\nDeepSeek Veredito Final (sem acao): <nenhuma resposta DeepSeek disponivel>")
            else:
                logger.info("Nenhum padrao detectado.")


if __name__ == "__main__":
    M = Monitor()

    # -------------------------------------------------
    #  NAO CHAMA O TRADERAI - evita a mensagem "Nao ha pergunta clara"
    # -------------------------------------------------
    # resp = M.AI.chat("Qual e a resposta ? ?.")
    # print("Qual a sua atuacao aqui ?:", resp.get("reply") if isinstance(resp, dict) else resp)

    M.rodar()
