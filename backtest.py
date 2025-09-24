# backtest.py
"""
Backtest dos padrões/heurísticas do robô com cálculo de MFE (Maximum Favorable Excursion).

Exemplo:
    python backtest.py --tfs M15 M30 H1 H4 --bars 12000 --min-prob 60 --dedupe 3

Saídas por timeframe e agregadas (GERAL):
- Contagem de sinais (GAIN / LOSS / PENDING) e WinRate
- MFE médio em pontos (todos) e MFE médio apenas nos LOSS
- Mesmas métricas por lado (COMPRA / VENDA)
- (Opcional) exporta CSV por timeframe com os trades (use --export)
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import MetaTrader5 as mt5

# Projeto
from connectors.mt5_connector import MT5Connector
from core.indicators import Indicators
from core.patterns import PatternDetector
from core.analyzer import Analyzer
from config import (
    ATIVO,
    TIMEFRAMES, TF_MINUTES,
    DETECTOR_PARAMS,
    EMA_PERIOD,
    MIN_PROB_CONSENSO,
    EVAL_BARS,
)


# ------------- Utils -------------

def detection_lookback_bars() -> int:
    """
    Tamanho mínimo da janela de detecção para que os detectores tenham contexto suficiente.
    Usa os parâmetros de lookback das estruturas (retângulo/triângulo/flags).
    """
    base = 80
    rect = int(DETECTOR_PARAMS.get("RECT_LOOKBACK", 40))
    tri  = int(DETECTOR_PARAMS.get("TRI_LOOKBACK", 40))
    flag = int(DETECTOR_PARAMS.get("FLAG_MAX_BARS", 12)) * 3
    return max(base, rect, tri, flag)


def evaluate_with_mfe(df_fw, side: str, entry: float, stop: float, target: float, point: float):
    """
    Avalia o resultado do trade e calcula MFE (Maximum Favorable Excursion) em pontos.
    df_fw: DataFrame de candles à frente (exclui o candle de detecção)
    side: "COMPRA" ou "VENDA"
    entry/stop/target: preços numéricos
    point: tamanho do "point" do símbolo (ex.: 0.01 em muitos brokers para XAUUSD)

    Retorna:
      outcome ∈ {"GAIN","LOSS","PENDING"},
      time_outcome (timestamp),
      mfe_pts (int, pontos favoráveis máximos).
    """
    max_fav = 0.0
    outcome = "PENDING"
    t_out = None

    for t, row in df_fw.iterrows():
        high = float(row["high"])
        low  = float(row["low"])

        if side == "COMPRA":
            fav_now = max(0.0, high - entry)   # quanto andou a favor
            hit_tp  = high >= target
            hit_sl  = low  <= stop
        else:
            fav_now = max(0.0, entry - low)
            hit_tp  = low  <= target
            hit_sl  = high >= stop

        if fav_now > max_fav:
            max_fav = fav_now

        # Ordem de verificação TP/SL: otimista (TP antes).
        # Para conservador, troque a ordem.
        if hit_tp:
            outcome = "GAIN"
            t_out = t
            break
        if hit_sl:
            outcome = "LOSS"
            t_out = t
            break

    if outcome == "PENDING" and len(df_fw) > 0:
        t_out = df_fw.index[-1]

    mfe_pts = int(round(max_fav / point)) if point > 0 else int(round(max_fav))
    return outcome, t_out, mfe_pts


def print_tf_report(tf_name: str, df_sig: pd.DataFrame):
    """
    Relatório textual por timeframe, com MFE médios.
    Espera um DataFrame com colunas:
      ['time','tf','pattern','side','prob','entry','stop','target','outcome','mfe_pts']
    """
    total = len(df_sig)
    gain = int((df_sig["outcome"] == "GAIN").sum())
    loss = int((df_sig["outcome"] == "LOSS").sum())
    pend = int((df_sig["outcome"] == "PENDING").sum())
    wr   = (gain / (gain + loss) * 100.0) if (gain + loss) > 0 else 0.0

    buy_df  = df_sig[df_sig["side"] == "COMPRA"]
    sell_df = df_sig[df_sig["side"] == "VENDA"]

    def wr_of(df_side):
        g = int((df_side["outcome"] == "GAIN").sum())
        l = int((df_side["outcome"] == "LOSS").sum())
        return (g / (g + l) * 100.0) if (g + l) > 0 else 0.0, g, l

    wr_buy, g_buy, l_buy   = wr_of(buy_df)
    wr_sell, g_sell, l_sell = wr_of(sell_df)

    def avg(series):
        return float(series.mean()) if len(series) else 0.0

    avg_mfe_all        = avg(df_sig["mfe_pts"])
    avg_mfe_loss_only  = avg(df_sig.loc[df_sig["outcome"] == "LOSS", "mfe_pts"])
    avg_mfe_buy        = avg(buy_df["mfe_pts"])
    avg_mfe_sell       = avg(sell_df["mfe_pts"])
    avg_mfe_loss_buy   = avg(buy_df.loc[buy_df["outcome"] == "LOSS", "mfe_pts"])
    avg_mfe_loss_sell  = avg(sell_df.loc[sell_df["outcome"] == "LOSS", "mfe_pts"])

    print(f"\n===== {tf_name} =====")
    print(f"Sinais: {total} | GAIN: {gain} | LOSS: {loss} | PENDING: {pend} | WinRate: {wr:.1f}%")
    print(f"- COMPRA: {len(buy_df)} sinais | G:{g_buy} L:{l_buy} | WR:{wr_buy:.1f}%")
    print(f"- VENDA:  {len(sell_df)} sinais | G:{g_sell} L:{l_sell} | WR:{wr_sell:.1f}%")
    print(f"MFE médio (pontos) — Todos: {avg_mfe_all:.1f} | Só LOSS: {avg_mfe_loss_only:.1f}")
    print(f"  · COMPRA: {avg_mfe_buy:.1f}  (Só LOSS: {avg_mfe_loss_buy:.1f})")
    print(f"  · VENDA : {avg_mfe_sell:.1f} (Só LOSS: {avg_mfe_loss_sell:.1f})")


# ------------- Núcleo do backtest -------------

def backtest_timeframe(mt5c: MT5Connector, tf_name: str, bars: int, min_prob: int, dedupe: int, export: bool):
    """
    Roda o backtest para um timeframe.
    - mt5c: conector MT5
    - tf_name: "M15", "M30", ...
    - bars: quantidade de candles a puxar
    - min_prob: probabilidade mínima (0..100) para aceitar um padrão
    - dedupe: distância mínima em candles entre sinais aceitos
    - export: se True, salva CSV 'bt_<tf>.csv' com os trades

    Retorna um DataFrame com colunas:
      ['time','tf','pattern','side','prob','entry','stop','target','outcome','mfe_pts']
    """
    tf = TIMEFRAMES[tf_name]
    df = mt5c.obter_candles(tf, bars)
    if df is None or df.empty:
        print(f"[BT] {tf_name}: sem dados.")
        return pd.DataFrame()

    # Indicadores
    df = Indicators.aplicar(df, ema_period=EMA_PERIOD)

    # Janela mínima de detecção
    L = detection_lookback_bars()

    # Point do símbolo (para converter MFE em pontos)
    si = mt5.symbol_info(ATIVO)
    point = float(si.point) if si and si.point else 0.01

    # Detecção + avaliação
    regs = []
    last_accept_idx = -10_000  # para "dedupe" de sinais
    # Garantir que haverá barras à frente para avaliação
    last_detect_index = len(df) - (EVAL_BARS + 1)
    if last_detect_index < L:
        print(f"[BT] {tf_name}: poucas barras para backtest. Aumente --bars.")
        return pd.DataFrame()

    # Laço
    for i in range(L, last_detect_index + 1):
        # janela para detecção termina no candle i (inclusive)
        sub = df.iloc[i - L: i + 1].copy()

        # detecta padrões no último candle da janela
        # (passa uma tolerância padrão para evitar None)
        tol = float(DETECTOR_PARAMS.get("RECT_BAND_TOL", 0.003))
        padroes = PatternDetector.detect_all(sub, tolerancia=tol, params=DETECTOR_PARAMS)
        if not padroes:
            continue

        # Seleciona *um* candle à frente para avaliar (janelão de EVAL_BARS)
        df_fw = df.iloc[i + 1: i + 1 + EVAL_BARS].copy()
        if df_fw.empty:
            continue

        # Sinal gerado no tempo do candle i (fechamento do sub)
        t_detect = sub.index[-1]

        # dedupe: espaçamento mínimo em candles
        if i - last_accept_idx < dedupe:
            continue

        # Para cada padrão detectado neste candle
        had_accepted = False
        for padrao in padroes:
            prob, sinal = Analyzer.calcular_probabilidade_por_padrao(
                sub, padrao
            )
            if prob < min_prob:
                continue
            if sinal not in ("COMPRA", "VENDA"):
                continue

            # monta níveis
            entry, stop, target, atr = Analyzer.montar_trade_levels(sub, sinal)

            # avalia + mede MFE
            outcome, t_out, mfe_pts = evaluate_with_mfe(df_fw, sinal, entry, stop, target, point)

            regs.append({
                "time": t_detect,
                "tf": tf_name,
                "pattern": padrao,
                "side": sinal,
                "prob": int(prob),
                "entry": float(entry),
                "stop": float(stop),
                "target": float(target),
                "outcome": outcome,
                "mfe_pts": int(mfe_pts),
            })
            had_accepted = True

        if had_accepted:
            last_accept_idx = i

    df_sig = pd.DataFrame(regs)
    if export and not df_sig.empty:
        fname = f"bt_{tf_name}_sinais.csv"
        df_sig.to_csv(fname, index=False)
        print(f"[BT] {tf_name}: exportado {fname} ({len(df_sig)} linhas)")

    return df_sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", nargs="+", default=["M15", "M30", "H1", "H4"], help="Lista de timeframes")
    ap.add_argument("--bars", type=int, default=12000, help="Qtd de candles por TF")
    ap.add_argument("--min-prob", type=int, default=MIN_PROB_CONSENSO, help="Prob mínima para aceitar o padrão (0..100)")
    ap.add_argument("--dedupe", type=int, default=3, help="Distância mínima (em candles) entre sinais aceitos")
    ap.add_argument("--export", action="store_true", help="Exportar CSV por timeframe")
    args = ap.parse_args()

    # Conexão MT5
    mt5c = MT5Connector()
    mt5c.conectar()

    all_frames = []
    for tf_name in args.tfs:
        print(f"[BT] Rodando {tf_name}…")
        df_sig = backtest_timeframe(mt5c, tf_name, args.bars, args.min_prob, args.dedupe, args.export)
        print_tf_report(tf_name, df_sig)
        all_frames.append(df_sig)

    # Agregado GERAL
    df_all = pd.concat(all_frames, ignore_index=True) if len(all_frames) else pd.DataFrame()
    if not df_all.empty:
        total = len(df_all)
        gain = int((df_all["outcome"] == "GAIN").sum())
        loss = int((df_all["outcome"] == "LOSS").sum())
        pend = int((df_all["outcome"] == "PENDING").sum())
        wr   = (gain / (gain + loss) * 100.0) if (gain + loss) > 0 else 0.0

        buy_df  = df_all[df_all["side"] == "COMPRA"]
        sell_df = df_all[df_all["side"] == "VENDA"]

        def wr_of(df_side):
            g = int((df_side["outcome"] == "GAIN").sum())
            l = int((df_side["outcome"] == "LOSS").sum())
            return (g / (g + l) * 100.0) if (g + l) > 0 else 0.0, g, l

        wr_buy, g_buy, l_buy   = wr_of(buy_df)
        wr_sell, g_sell, l_sell = wr_of(sell_df)

        def avg(series):
            return float(series.mean()) if len(series) else 0.0

        avg_mfe_all        = avg(df_all["mfe_pts"])
        avg_mfe_loss_only  = avg(df_all.loc[df_all["outcome"] == "LOSS", "mfe_pts"])
        avg_mfe_buy        = avg(buy_df["mfe_pts"])
        avg_mfe_sell       = avg(sell_df["mfe_pts"])
        avg_mfe_loss_buy   = avg(buy_df.loc[buy_df["outcome"] == "LOSS", "mfe_pts"])
        avg_mfe_loss_sell  = avg(sell_df.loc[sell_df["outcome"] == "LOSS", "mfe_pts"])

        print(f"\n===== GERAL =====")
        print(f"Sinais: {total} | GAIN: {gain} | LOSS: {loss} | PENDING: {pend} | WinRate: {wr:.1f}%")
        print(f"- COMPRA: {len(buy_df)} sinais | G:{g_buy} L:{l_buy} | WR:{wr_buy:.1f}%")
        print(f"- VENDA:  {len(sell_df)} sinais | G:{g_sell} L:{l_sell} | WR:{wr_sell:.1f}%")
        print(f"MFE médio (pontos) — Todos: {avg_mfe_all:.1f} | Só LOSS: {avg_mfe_loss_only:.1f}")
        print(f"  · COMPRA: {avg_mfe_buy:.1f}  (Só LOSS: {avg_mfe_loss_buy:.1f})")
        print(f"  · VENDA : {avg_mfe_sell:.1f} (Só LOSS: {avg_mfe_loss_sell:.1f})")
    else:
        print("\n===== GERAL =====\nSem sinais.")


if __name__ == "__main__":
    main()
