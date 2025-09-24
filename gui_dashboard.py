import os
import sys
import threading
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox

# Dependências do seu projeto
try:
    import pandas as pd
except Exception as e:
    print("Pandas é necessário para o painel. Instale com: pip install pandas")
    raise

try:
    import MetaTrader5 as mt5
except Exception as e:
    mt5 = None

# Importa sua configuração e classes
try:
    import config
    from core.verdict_trader import VerdictTraderConfig
    from core.patterns import PatternDetector
except Exception as e:
    print("Erro ao importar módulos do projeto:", e)
    traceback.print_exc()
    sys.exit(1)


def _safe_read_csv(path: str, n: int = 300):
    """Lê CSV com tolerância a arquivo em uso. Retorna DataFrame (pode ser vazio)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Normaliza possíveis colunas esperadas
        if "detect_time" in df.columns:
            try:
                df["detect_time"] = pd.to_datetime(df["detect_time"])
            except Exception:
                pass
        # Mostra somente os mais recentes
        if len(df) > n:
            df = df.tail(n)
        return df
    except Exception:
        # tentativa com engine diferente
        try:
            df = pd.read_csv(path, engine="python")
            if "detect_time" in df.columns:
                try:
                    df["detect_time"] = pd.to_datetime(df["detect_time"])
                except Exception:
                    pass
            if len(df) > n:
                df = df.tail(n)
            return df
        except Exception:
            return pd.DataFrame()


class Dashboard(tk.Tk):
    def __init__(self, refresh_ms: int = 2000):
        super().__init__()
        self.title("Megazord — Painel do Robô")
        self.geometry("1180x760")
        self.minsize(1000, 680)

        self.refresh_ms = refresh_ms
        self._mt5_inited = False
        self._symbol = None
        self._magic = None

        # Carrega confs
        self._load_configs()

        # UI
        self._build_layout()

        # Inicializa MT5 para leitura (se disponível)
        self._init_mt5()

        # Primeiro refresh
        self.after(400, self.refresh_all)

    # ---------------- Config & MT5 ----------------

    def _load_configs(self):
        # Configs do trader (SL/TP/Trail e etc.)
        vt_cfg = VerdictTraderConfig()
        self._symbol = vt_cfg.symbol
        self._magic = vt_cfg.magic

        self.cfg_overview = {
            "DEEPSEEK": bool(getattr(config, "DEEPSEEK_API_KEY", "")),
            "Ativo/Símbolo": vt_cfg.symbol,
            "Lote (base)": vt_cfg.lot,
            "Desvio (pts)": vt_cfg.deviation_points,
            "Magic": vt_cfg.magic,
            "Comentário": vt_cfg.comment,
            "BUY ≥": vt_cfg.buy_threshold,
            "SELL ≤": vt_cfg.sell_threshold,
            "SL (pts)": vt_cfg.sl_points,
            "TP (pts)": vt_cfg.tp_points,
            "Trailing (pts)": vt_cfg.trail_points,
            "Timeframes": ", ".join(config.TIMEFRAMES.keys()),
            "Pesos": ", ".join(f"{k}:{v}" for k, v in config.PESOS_TIMEFRAMES.items()),
            "Min Prob. Consenso": config.MIN_PROB_CONSENSO,
            "Min Score Trade": config.MIN_SCORE_TRADE,
            "Intervalo (min)": config.INTERVALO_MINUTOS,
            "QTD Candles": config.QTD_CANDLES,
        }
        self.enabled_patterns = getattr(config, "ENABLED_PATTERNS", PatternDetector.listar_padroes())
        self.signal_csv = getattr(config, "SIGNAL_LOG_CSV", "signals_log.csv")

    def _init_mt5(self):
        if mt5 is None:
            self.mt5_status_var.set("MetaTrader5: módulo não disponível")
            return
        try:
            ok = mt5.initialize()
            self._mt5_inited = bool(ok)
            if ok:
                term_info = mt5.terminal_info()
                self.mt5_status_var.set(f"MetaTrader5: conectado | Versão {getattr(term_info,'build', '?')}")
            else:
                self.mt5_status_var.set(f"MetaTrader5: falha initialize() {mt5.last_error()}")
        except Exception as e:
            self.mt5_status_var.set(f"MetaTrader5: erro init — {e}")

    # ---------------- UI ----------------

    def _build_layout(self):
        # Top bar — status + controles
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)

        self.mt5_status_var = tk.StringVar(value="MetaTrader5: (inicializando)")
        ttk.Label(top, textvariable=self.mt5_status_var).pack(side=tk.LEFT)

        ttk.Label(top, text=" | Atualização (ms): ").pack(side=tk.LEFT, padx=(12, 2))
        self.refresh_entry = ttk.Entry(top, width=8)
        self.refresh_entry.insert(0, str(self.refresh_ms))
        self.refresh_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(top, text="Aplicar", command=self._apply_refresh).pack(side=tk.LEFT)
        ttk.Button(top, text="Atualizar agora", command=self.refresh_all).pack(side=tk.LEFT, padx=(8, 0))

        # Tabs
        tabs = ttk.Notebook(self)
        tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Overview
        self.tab_overview = ttk.Frame(tabs)
        tabs.add(self.tab_overview, text="Overview")

        # Sinais
        self.tab_signals = ttk.Frame(tabs)
        tabs.add(self.tab_signals, text="Sinais (ao vivo)")

        # Posições
        self.tab_positions = ttk.Frame(tabs)
        tabs.add(self.tab_positions, text="Posições")

        self._build_overview_tab()
        self._build_signals_tab()
        self._build_positions_tab()

        # Rodapé
        bottom = ttk.Frame(self, padding=(10, 6))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Pronto")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)

    def _build_overview_tab(self):
        left = ttk.Frame(self.tab_overview, padding=(10, 10))
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(self.tab_overview, padding=(10, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configs principais
        g1 = ttk.LabelFrame(left, text="Parâmetros / Execução")
        g1.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        tree_cfg = ttk.Treeview(g1, columns=("k", "v"), show="headings", height=16)
        tree_cfg.heading("k", text="Parâmetro")
        tree_cfg.heading("v", text="Valor")
        tree_cfg.column("k", width=220, anchor=tk.W)
        tree_cfg.column("v", width=260, anchor=tk.W)
        tree_cfg.pack(fill=tk.BOTH, expand=True)
        self.tree_cfg = tree_cfg

        # Padrões habilitados
        g2 = ttk.LabelFrame(right, text="Padrões habilitados")
        g2.pack(fill=tk.BOTH, expand=True)

        self.patterns_text = tk.Text(g2, height=16, wrap=tk.WORD)
        self.patterns_text.pack(fill=tk.BOTH, expand=True)

        # Carrega conteúdo estático
        self._populate_overview()

    def _populate_overview(self):
        # params
        for k, v in self.cfg_overview.items():
            self.tree_cfg.insert("", tk.END, values=(k, v))
        # padrões
        self.patterns_text.delete("1.0", tk.END)
        if self.enabled_patterns:
            self.patterns_text.insert(tk.END, "• " + "\n• ".join(self.enabled_patterns))
        else:
            self.patterns_text.insert(tk.END, "(Nenhum padrão habilitado)")

    def _build_signals_tab(self):
        topbar = ttk.Frame(self.tab_signals)
        topbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(topbar, text=f"Arquivo de sinais: {self.signal_csv}").pack(side=tk.LEFT)

        ttk.Label(topbar, text=" | Mostrar últimos: ").pack(side=tk.LEFT, padx=(10, 2))
        self.last_n_var = tk.StringVar(value="150")
        ttk.Entry(topbar, textvariable=self.last_n_var, width=6).pack(side=tk.LEFT)

        ttk.Button(topbar, text="Atualizar", command=self.refresh_signals).pack(side=tk.LEFT, padx=(8, 0))

        # Tabela
        cols = ("detect_time", "timeframe", "pattern", "signal", "prob", "price", "entry", "stop", "target", "atr", "status")
        tree = ttk.Treeview(self.tab_signals, columns=cols, show="headings", height=24)
        for c in cols:
            tree.heading(c, text=c)
            w = 110 if c in ("pattern",) else 90
            if c in ("detect_time", "pattern"):
                w = 140
            if c in ("entry", "stop", "target", "atr"):
                w = 80
            tree.column(c, width=w, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.tree_signals = tree

    def _build_positions_tab(self):
        topbar = ttk.Frame(self.tab_positions)
        topbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(topbar, text=f"Símbolo: {self._symbol} | Magic: {self._magic}").pack(side=tk.LEFT)
        ttk.Button(topbar, text="Atualizar", command=self.refresh_positions).pack(side=tk.LEFT, padx=(10, 0))

        cols = ("ticket", "type", "volume", "price_open", "sl", "tp", "profit", "swap", "time")
        tree = ttk.Treeview(self.tab_positions, columns=cols, show="headings", height=22)
        names = {
            "ticket": "Ticket", "type": "Tipo", "volume": "Volume", "price_open": "Preço Abert.",
            "sl": "SL", "tp": "TP", "profit": "Profit", "swap": "Swap", "time": "Tempo"
        }
        widths = {"ticket": 100, "type": 70, "volume": 80, "price_open": 100, "sl": 100, "tp": 100, "profit": 100, "swap": 80, "time": 140}
        for c in cols:
            tree.heading(c, text=names[c])
            tree.column(c, width=widths[c], anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.tree_pos = tree

    # ---------------- Refresh Loop ----------------

    def _apply_refresh(self):
        try:
            ms = int(self.refresh_entry.get().strip())
            ms = max(500, min(ms, 10000))
            self.refresh_ms = ms
            self.status_var.set(f"Intervalo de atualização: {ms} ms")
        except Exception:
            messagebox.showwarning("Valor inválido", "Digite um número inteiro em milissegundos.")

    def refresh_all(self):
        self.refresh_signals()
        self.refresh_positions()
        self.after(self.refresh_ms, self.refresh_all)

    def refresh_signals(self):
        try:
            n = int(self.last_n_var.get())
        except Exception:
            n = 150
        df = _safe_read_csv(self.signal_csv, n=n)
        # Apaga linhas atuais
        for i in self.tree_signals.get_children():
            self.tree_signals.delete(i)
        if df.empty:
            self.status_var.set("Nenhum sinal encontrado (signals_log.csv).")
            return

        # Normaliza nomes
        # Esperado no salvar_sinal: detect_time, timeframe, pattern, signal, prob, price, entry, stop, target, atr, status...
        remap = {
            "timeframe": "timeframe", "pattern": "pattern", "signal": "signal",
            "prob": "prob", "price": "price", "entry": "entry", "stop": "stop", "target": "target",
            "atr": "atr", "status": "status", "detect_time": "detect_time"
        }
        for k in remap:
            if k not in df.columns and k.capitalize() in df.columns:
                df[k] = df[k.capitalize()]

        # Ordena por detect_time se existir
        if "detect_time" in df.columns:
            try:
                df = df.sort_values("detect_time", ascending=False)
            except Exception:
                pass

        for _, row in df.iterrows():
            vals = (
                str(row.get("detect_time", ""))[:19],
                str(row.get("timeframe", "")),
                str(row.get("pattern", "")),
                str(row.get("signal", "")),
                str(row.get("prob", "")),
                str(row.get("price", "")),
                str(row.get("entry", "")),
                str(row.get("stop", "")),
                str(row.get("target", "")),
                str(row.get("atr", "")),
                str(row.get("status", "")),
            )
            self.tree_signals.insert("", tk.END, values=vals)
        self.status_var.set(f"Sinais atualizados • {datetime.now().strftime('%H:%M:%S')}")

    def refresh_positions(self):
        if not self._mt5_inited or mt5 is None:
            return
        poss = mt5.positions_get(symbol=self._symbol) or []

        # limpa
        for i in self.tree_pos.get_children():
            self.tree_pos.delete(i)

        count = 0
        for p in poss:
            try:
                if getattr(p, "magic", 0) != self._magic:
                    continue
                typ = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
                vals = (
                    str(getattr(p, "ticket", "")),
                    typ,
                    f"{float(getattr(p, 'volume', 0.0)):.2f}",
                    f"{float(getattr(p, 'price_open', 0.0)):.2f}",
                    f"{float(getattr(p, 'sl', 0.0)):.2f}",
                    f"{float(getattr(p, 'tp', 0.0)):.2f}",
                    f"{float(getattr(p, 'profit', 0.0)):.2f}",
                    f"{float(getattr(p, 'swap', 0.0)):.2f}",
                    datetime.fromtimestamp(getattr(p, 'time', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                )
                self.tree_pos.insert("", tk.END, values=vals)
                count += 1
            except Exception:
                continue

        self.status_var.set(f"Posições abertas (símbolo={self._symbol}, magic={self._magic}): {count}")


if __name__ == "__main__":
    # Tema ttk (opcional)
    try:
        import sv_ttk  # pip install sv-ttk
        sv_ttk.set_theme("dark")
    except Exception:
        pass

    app = Dashboard(refresh_ms=2000)
    app.mainloop()
