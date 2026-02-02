#!/usr/bin/env python3
"""
Atualiza a chave "date" em daily_profit_data.json para o dia seguinte
todos os dias às 18:00 (America/Sao_Paulo), acordando apenas na virada da hora.

Exemplo:
  {"date":"2025-12-24", ...} -> {"date":"2025-12-25", ...}

Requisitos:
  - Python 3.9+ (zoneinfo)
"""

import os
import json
import time
from datetime import datetime, timedelta, date as date_cls
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Sao_Paulo")
TARGET_HOUR = 18
FILENAME = "daily_profit_data.json"


def log(msg: str) -> None:
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} - America/Sao_Paulo] {msg}", flush=True)


def file_path_in_same_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, FILENAME)


def sleep_until_next_full_hour() -> None:
    """Dorme até a próxima virada de hora (HH:00:00) no fuso de Brasília."""
    now = datetime.now(TZ)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    seconds = (next_hour - now).total_seconds()
    time.sleep(max(1, int(seconds)))


def parse_yyyy_mm_dd(s: str) -> date_cls:
    return datetime.strptime(s, "%Y-%m-%d").date()


def bump_date_in_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    # Lê JSON
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError("Arquivo está vazio. Esperava um JSON com a chave 'date'.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido no arquivo: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("JSON raiz precisa ser um objeto (dict).")

    if "date" not in data:
        raise KeyError("Chave 'date' não existe no JSON.")

    old_date_str = str(data["date"]).strip()
    old_date = parse_yyyy_mm_dd(old_date_str)
    new_date = old_date + timedelta(days=1)
    data["date"] = new_date.isoformat()

    # Salva (mantém legível)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(", ", ": "))

    log(f"OK: date atualizada de {old_date.isoformat()} -> {new_date.isoformat()}")


def main() -> None:
    path = file_path_in_same_dir()
    log("Monitor iniciado. Vou acordar só na virada de cada hora e executar às 18:00.")
    log(f"Arquivo alvo: {path}")

    last_run_date = None  # evita rodar 2x no mesmo dia (se reiniciar por volta das 18:00)

    while True:
        sleep_until_next_full_hour()

        now = datetime.now(TZ)
        if now.hour == TARGET_HOUR and now.minute == 0:
            if last_run_date == now.date():
                continue

            try:
                bump_date_in_file(path)
                last_run_date = now.date()
            except Exception as e:
                log(f"ERRO: {e!r}")
        else:
            log(f"Check horário: {now.strftime('%H:%M')} (nada a fazer).")


if __name__ == "__main__":
    main()
