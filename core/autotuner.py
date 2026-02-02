"""AutoTuner
Gerencia log de sinais, avalia resultados após N candles e recalibra parâmetros.
Uso:
    tuner = AutoTuner()
    tuner.salvar_sinal(registro)
    tuner.avaliar_pendentes(mt5c)
    params = tuner.ajustar_parametros()
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import (
    SIGNAL_LOG_CSV, PARAM_LOG_CSV, TUNING_INTERVAL_MIN, TUNING_MIN_SIGNALS,
    TF_MINUTES, RSI_SOBRECOMPRA, RSI_SOBREVENDIDO, EMA_PERIOD, TOLERANCIA_PADRAO,
    ATR_MULT_STOP, RR_MULT, EVAL_BARS
)

from core.analyzer import Analyzer
from pathlib import Path
# ... resto dos imports


class AutoTuner:
    def __init__(self, signal_log: str = SIGNAL_LOG_CSV, param_log: str = PARAM_LOG_CSV):
        # Garante que o diretório dos arquivos existe (mesmo se o caminho for só o nome do arquivo)
        Path(signal_log).parent.mkdir(parents=True, exist_ok=True)
        Path(param_log).parent.mkdir(parents=True, exist_ok=True)

        self.signal_log = signal_log
        self.param_log  = param_log
        self.params = {
            "RSI_SOBRECOMPRA": RSI_SOBRECOMPRA,
            "RSI_SOBREVENDIDO": RSI_SOBREVENDIDO,
            "EMA_PERIOD": EMA_PERIOD,
            "TOLERANCIA": TOLERANCIA_PADRAO,
            "ATR_MULT_STOP": ATR_MULT_STOP,
            "RR_MULT": RR_MULT,
        }
        self._last_tuning = datetime.utcnow() - timedelta(minutes=TUNING_INTERVAL_MIN + 1)

    def salvar_sinal(self, registro: dict) -> None:
        """Grava um sinal (PENDING) no CSV para posterior avaliação/ajuste."""
        row = pd.DataFrame([registro])
        if not os.path.isfile(self.signal_log):
            row.to_csv(self.signal_log, index=False)
        else:
            row.to_csv(self.signal_log, mode='a', header=False, index=False)

    # =========================
    # LEITURA ROBUSTA DO CSV
    # =========================
    def _carregar_sinais(self):
        """Lê o CSV de sinais de forma tolerante e com tipos consistentes."""
        if not os.path.exists(self.signal_log):
            return pd.DataFrame()

        try:
            df = pd.read_csv(
                self.signal_log,
                parse_dates=["detect_time", "evaluated_at"],
                dtype={
                    "symbol": "string",
                    "timeframe": "string",
                    "pattern": "string",
                    "status": "string",
                    "entry": "float64",
                    "stop": "float64",
                    "target": "float64",
                },
                na_values=["", "NaN", "null", None],
                low_memory=False
            )
        except Exception as e1:
            try:
                # Fallback extremo
                df = pd.read_csv(self.signal_log, engine="python")
            except Exception as e2:
                print(f"[AutoTuner] Falha ao carregar {self.signal_log}: {e1} / {e2}")
                return pd.DataFrame()

        # =========================
        # NORMALIZAÇÕES DEFENSIVAS
        # =========================
        for col in ["detect_time", "evaluated_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "status" in df.columns:
            df["status"] = (
                df["status"]
                .astype("string")
                .str.upper()
                .str.strip()
            )

        if "pattern" in df.columns:
            df["pattern"] = (
                df["pattern"]
                .astype("string")
                .str.strip()
            )

        # Remove registros estruturalmente inválidos (não interfere nos válidos)
        df = df.dropna(subset=["status", "entry", "stop", "target"], how="any")

        return df

    def avaliar_pendentes(self, mt5c) -> int:
        """Percorre sinais PENDING e tenta classificá-los como GAIN/LOSS após N candles."""
        df = self._carregar_sinais()
        if df.empty:
            return 0

        atualizados = 0
        pend = df[df["status"] == "PENDING"]

        for idx, r in pend.iterrows():
            tf_name = r["timeframe"]

            # segurança: sem timestamp não dá pra avaliar
            if pd.isna(r.get("detect_time")):
                continue

            tf_minutes = TF_MINUTES.get(tf_name, 5)
            min_dt = r["detect_time"] + timedelta(minutes=tf_minutes * EVAL_BARS)
            if datetime.utcnow() < min_dt:
                continue

            entry = float(r["entry"])
            stop = float(r["stop"])
            target = float(r["target"])

            status, when = Analyzer.avaliar_trade(
                mt5c,
                tf_name,
                r["detect_time"],
                entry,
                stop,
                target,
                eval_bars=EVAL_BARS
            )

            if status != "PENDING":
                df.at[idx, "status"] = status
                df.at[idx, "evaluated_at"] = when if when is not None else datetime.utcnow()
                atualizados += 1

        if atualizados:
            df.to_csv(self.signal_log, index=False)

        return atualizados

    def _registrar_parametros(self, metrics: dict) -> None:
        """Salva snapshot dos parâmetros + métricas de acurácia atuais."""
        row = {"timestamp": datetime.utcnow(), **self.params, **metrics}
        df = pd.DataFrame([row])
        if not os.path.isfile(self.param_log):
            df.to_csv(self.param_log, index=False)
        else:
            df.to_csv(self.param_log, mode='a', header=False, index=False)

    def ajustar_parametros(self) -> dict:
        """Recalibra parâmetros a partir da acurácia recente (thresholds simples)."""
        if (datetime.utcnow() - self._last_tuning).total_seconds() < TUNING_INTERVAL_MIN * 60:
            return self.params

        df = self._carregar_sinais()
        if df.empty or "status" not in df:
            return self.params

        df_eval = df[df["status"].isin(["GAIN", "LOSS"])]
        if len(df_eval) < TUNING_MIN_SIGNALS:
            return self.params

        acc_by_pattern = (
            df_eval
            .groupby("pattern")["status"]
            .apply(lambda x: (x == "GAIN").mean())
            .to_dict()
        )

        acc_global = (df_eval["status"] == "GAIN").mean()

        # =========================
        # REGRAS DE AJUSTE
        # =========================
        if acc_by_pattern.get("Double Bottom", 1.0) < 0.55:
            self.params["RSI_SOBREVENDIDO"] = min(45, self.params["RSI_SOBREVENDIDO"] + 3)
            self.params["TOLERANCIA"]       = min(0.0025, self.params["TOLERANCIA"] + 0.0002)

        if acc_by_pattern.get("Double Top", 1.0) < 0.55:
            self.params["RSI_SOBRECOMPRA"] = max(55, self.params["RSI_SOBRECOMPRA"] - 3)
            self.params["TOLERANCIA"]      = min(0.0025, self.params["TOLERANCIA"] + 0.0002)

        if acc_global < 0.55:
            self.params["ATR_MULT_STOP"] = min(2.0, self.params["ATR_MULT_STOP"] + 0.1)
            self.params["RR_MULT"]       = max(1.4, self.params["RR_MULT"] - 0.1)

        metrics = {
            "acc_double_bottom": acc_by_pattern.get("Double Bottom", np.nan),
            "acc_double_top":    acc_by_pattern.get("Double Top", np.nan),
            "acc_global":        acc_global,
            "samples":           len(df_eval)
        }

        self._registrar_parametros(metrics)
        self._last_tuning = datetime.utcnow()
        return self.params
