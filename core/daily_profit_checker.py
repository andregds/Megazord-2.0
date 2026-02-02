# core/daily_profit_checker.py
import MetaTrader5 as mt5
import datetime
import os
import json
import logging
import time
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DAILY_PROFIT_FILE = "daily_profit_data.json"
PROFIT_TARGET = 10000.00  # USD

ACCOUNT_INFO_RETRIES = 12
ACCOUNT_INFO_DELAY_SEC = 0.5


class DailyProfitChecker:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.current_mt5_date: Optional[datetime.date] = None

        self.daily_profit: float = 0.0
        self.initial_daily_balance: Optional[float] = None  # baseline pode começar como None
        self.initial_balance_set: bool = False

        self._ensure_mt5_connection()

        self.current_mt5_date = self._get_mt5_server_date()
        if self.current_mt5_date is None:
            logger.error("Não foi possível obter a data do servidor MT5 na inicialização. Usando data do sistema.")
            self.current_mt5_date = datetime.date.today()

        self._load_daily_state()

    # -----------------------------
    # MT5 helpers
    # -----------------------------
    def _ensure_mt5_connection(self) -> bool:
        """
        Inicializa o MT5 (se necessário). Não grava baseline aqui.
        Retorna True se o terminal inicializou; account_info pode demorar alguns ms para ficar disponível.
        """
        if mt5.initialize():
            return True

        logger.warning("MT5 não inicializado. Tentando inicializar novamente...")
        for i in range(1, 4):
            time.sleep(1)
            if mt5.initialize():
                logger.info(f"MT5 inicializado com sucesso após {i} tentativa(s).")
                return True

        logger.critical("Não foi possível inicializar MT5 após múltiplas tentativas.")
        return False

    def _get_account_balance_with_retry(self) -> Optional[float]:
        """
        Pega account_info().balance com retries.
        Retorna None se não conseguir ler a conta.
        """
        self._ensure_mt5_connection()

        for _ in range(ACCOUNT_INFO_RETRIES):
            info = mt5.account_info()
            if info:
                return float(info.balance)
            time.sleep(ACCOUNT_INFO_DELAY_SEC)

        return None

    def _get_mt5_server_date(self) -> Optional[datetime.date]:
        if not self._ensure_mt5_connection():
            logger.error("MT5 não conectado para obter a data do servidor.")
            return None

        # garante símbolo selecionado (evita tick None em alguns terminais)
        try:
            mt5.symbol_select(self.symbol, True)
        except Exception:
            pass

        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            return datetime.datetime.fromtimestamp(tick.time).date()

        logger.error(f"Não foi possível obter tick do símbolo {self.symbol}. Usando data do sistema.")
        return datetime.date.today()

    # -----------------------------
    # Persistência
    # -----------------------------
    def _save_daily_state(self):
        data = {
            "date": self.current_mt5_date.strftime("%Y-%m-%d"),
            "profit": float(self.daily_profit),
            "initial_balance": (float(self.initial_daily_balance) if self.initial_daily_balance is not None else None),
            "initial_balance_set": bool(self.initial_balance_set),
        }
        with open(DAILY_PROFIT_FILE, "w") as f:
            json.dump(data, f)

    def _reset_daily_state(self):
        """
        Novo dia (ou baseline inválido): zera lucro e deixa baseline para ser setado
        assim que conseguirmos ler o saldo da conta.
        """
        self.daily_profit = 0.0
        self.initial_daily_balance = None
        self.initial_balance_set = False
        self._save_daily_state()

    def _try_set_baseline_now(self) -> bool:
        """
        Define baseline = saldo atual (no momento que o robô conseguiu ler a conta).
        Retorna True se setou.
        """
        bal = self._get_account_balance_with_retry()
        if bal is None:
            logger.warning("Ainda não consegui ler account_info().balance. Baseline não definido (por enquanto).")
            return False

        self.initial_daily_balance = bal
        self.initial_balance_set = True
        self.daily_profit = 0.0
        self._save_daily_state()
        logger.info(f"✅ Baseline do dia setado: ${self.initial_daily_balance:.2f} (Data MT5: {self.current_mt5_date})")
        return True

    def _load_daily_state(self):
        mt5_today = self._get_mt5_server_date() or datetime.date.today()
        self.current_mt5_date = mt5_today

        if not os.path.exists(DAILY_PROFIT_FILE):
            logger.info(f"Arquivo {DAILY_PROFIT_FILE} não encontrado. Iniciando estado diário.")
            self._reset_daily_state()
            # tenta setar baseline já na largada
            self._try_set_baseline_now()
            return

        try:
            with open(DAILY_PROFIT_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            logger.warning(f"Erro ao ler {DAILY_PROFIT_FILE}. Resetando estado diário.")
            self._reset_daily_state()
            self._try_set_baseline_now()
            return

        last_date_str = data.get("date")
        if not last_date_str:
            logger.warning(f"{DAILY_PROFIT_FILE} sem 'date'. Resetando estado diário.")
            self._reset_daily_state()
            self._try_set_baseline_now()
            return

        try:
            last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
        except Exception:
            logger.warning(f"{DAILY_PROFIT_FILE} com 'date' inválida. Resetando estado diário.")
            self._reset_daily_state()
            self._try_set_baseline_now()
            return

        if last_date != self.current_mt5_date:
            logger.info(f"Novo dia do MT5 detectado ({self.current_mt5_date}). Resetando estado diário.")
            self._reset_daily_state()
            self._try_set_baseline_now()
            return

        # Mesmo dia: carrega estado
        self.daily_profit = float(data.get("profit", 0.0) or 0.0)

        ib = data.get("initial_balance", None)
        self.initial_daily_balance = float(ib) if ib is not None else None

        self.initial_balance_set = bool(data.get("initial_balance_set", self.initial_daily_balance is not None))

        logger.info(
            f"Estado diário carregado para {self.current_mt5_date}: "
            f"Lucro ${self.daily_profit:.2f}, "
            f"Saldo Inicial ${0.0 if self.initial_daily_balance is None else self.initial_daily_balance:.2f}"
        )

        # ✅ Auto-correção: se baseline do arquivo tá inválido (0/None), define baseline agora (saldo atual)
        if (not self.initial_balance_set) or (self.initial_daily_balance is None) or (self.initial_daily_balance <= 0.0):
            logger.warning("Baseline inválido no arquivo (0/None). Re-basing para o saldo atual e zerando lucro do dia.")
            self._reset_daily_state()
            self._try_set_baseline_now()

    # -----------------------------
    # Checagem
    # -----------------------------
    def update_and_check_profit(self) -> bool:
        current_mt5_date_check = self._get_mt5_server_date() or self.current_mt5_date

        # Virou o dia: reseta e tenta setar baseline
        if current_mt5_date_check != self.current_mt5_date:
            self.current_mt5_date = current_mt5_date_check
            self._reset_daily_state()
            self._try_set_baseline_now()
            logger.info(f"Novo dia do MT5 detectado ({self.current_mt5_date}). Estado diário resetado.")

        # Se baseline ainda não foi setado, tenta agora e NÃO bloqueia por isso
        if not self.initial_balance_set or self.initial_daily_balance is None:
            self._try_set_baseline_now()
            # se ainda não conseguiu, deixa operar (pra não travar por bug de leitura)
            if not self.initial_balance_set or self.initial_daily_balance is None:
                logger.info("Baseline ainda não definido; não vou bloquear operações por enquanto.")
                return False

        current_balance = self._get_account_balance_with_retry()
        if current_balance is None:
            logger.warning("Não consegui ler saldo atual. Mantendo último lucro calculado e não bloqueando.")
            return False

        self.daily_profit = float(current_balance - float(self.initial_daily_balance))
        self._save_daily_state()

        if self.daily_profit >= PROFIT_TARGET:
            logger.warning(
                f"Limite de lucro diário de ${PROFIT_TARGET:.2f} atingido para {self.symbol} "
                f"(Data MT5: {self.current_mt5_date})! Lucro atual: ${self.daily_profit:.2f}. "
                f"Novas ordens serão bloqueadas até o próximo dia."
            )
            return True

        logger.info(
            f"Lucro diário atual para {self.symbol} (Data MT5: {self.current_mt5_date}): "
            f"${self.daily_profit:.2f}. Limite de ${PROFIT_TARGET:.2f} ainda não atingido."
        )
        return False
