# trader_ai.py
from __future__ import annotations
import os, json, re, requests
from datetime import datetime, timezone
from typing import Any, Callable, Optional
try:
    from config import OLLAMA_API_BASE_URL, OLLAMA_MODEL_NAME
except Exception:
    OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL")
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")

class TraderAI:
    def __init__(self,
                 trader: Any,
                 send_order_fn: Callable[..., Any],
                 sanitize_fn: Optional[Callable[..., Any]] = None,
                 symbol: str = "XAUUSD",
                 default_near_pct: float = 0.0001,
                 default_invert_flag: int = 0):
        self.trader = trader
        self.send_order_fn = send_order_fn
        self.sanitize_fn = sanitize_fn
        self.symbol = symbol
        self.default_near_pct = default_near_pct
        self.default_invert_flag = default_invert_flag

        # Usa Ollama local quando disponível
        self.ollama_base = OLLAMA_API_BASE_URL or None
        self.model = OLLAMA_MODEL_NAME or os.getenv("OLLAMA_MODEL_NAME", "deepseek-chat")
        self.chat_path = "./calibration/trader_chat.jsonl"
        self.notes_path = "./calibration/coach_notes.jsonl"
        os.makedirs("./calibration", exist_ok=True)

    # ===== util =====
    def _utcnow(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_jsonl(self, path: str, obj: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ===== ensinar/memos =====
    def teach(self, text: str, tags: Optional[list[str]] = None) -> dict:
        note = {"ts": self._utcnow(), "tags": tags or [], "text": text.strip()}
        self._append_jsonl(self.notes_path, note)
        return {"ok": True, "stored": note}

    def _notes_summary(self, max_chars=600) -> str:
        if not os.path.exists(self.notes_path):
            return ""
        try:
            lines = open(self.notes_path, "r", encoding="utf-8").read().strip().splitlines()[-20:]
            items = [json.loads(x).get("text", "") for x in lines if x.strip()]
            joined = " | ".join([x for x in items if x])
            return (joined[:max_chars] + "…") if len(joined) > max_chars else joined
        except Exception:
            return ""

    # ===== chat =====
    def chat(self, text: str) -> dict:
        sys_msg = ("Você é o trader do robô Megazord Gold Premium (XAUUSD). "
                   "Responda curto, prático e em PT-BR, focado em execução/risco. sempre a favor da tendência")
        notes = self._notes_summary()
        if notes:
            sys_msg += f"\nMemorandos do operador: {notes}"

        if not self.ollama_base:
            reply = "IA externa não configurada (OLLAMA_API_BASE_URL ausente). Anotei sua mensagem."
            self._append_jsonl(self.chat_path, {"ts": self._utcnow(), "user": text, "assistant": reply})
            return {"ok": True, "reply": reply, "used_model": None}

        try:
            # Construir prompt para Ollama
            prompt = f"[SYSTEM] {sys_msg}\n\n[USER] {text}"
            body = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
            try:
                # Aumenta timeout para permitir respostas mais lentas do Ollama local (antes 4.5s)
                r = requests.post(self.ollama_base, headers={"Content-Type": "application/json"}, json=body, timeout=45.0)
                r.raise_for_status()
                data = r.json()
                # tenta extrair texto em várias chaves possíveis
                if isinstance(data, dict):
                    if "result" in data and isinstance(data["result"], dict):
                        reply = data["result"].get("output") or data["result"].get("text") or str(data["result"])
                    elif "output" in data:
                        reply = data.get("output")
                    elif "response" in data:
                        reply = data.get("response")
                    else:
                        # fallback para raw text
                        reply = json.dumps(data, ensure_ascii=False)
                else:
                    reply = str(data)
                if isinstance(reply, list):
                    reply = "\n".join(str(x) for x in reply)
                reply = str(reply).strip()
            except requests.exceptions.Timeout:
                reply = "(Falha: Timeout ao conectar com Ollama local)"
            except requests.exceptions.ConnectionError:
                reply = "(Falha: Não foi possível conectar com Ollama local)"
            except json.JSONDecodeError:
                reply = "(Falha: Resposta inválida do Ollama local)"
            except Exception as e:
                reply = f"(Falha no Ollama: {e})"
        except Exception as e:
            reply = f"(Falha no DeepSeek: {e})"

        self._append_jsonl(self.chat_path, {"ts": self._utcnow(), "user": text, "assistant": reply})
        return {"ok": True, "reply": reply, "used_model": self.model}

    # ===== parsing simples de ordens no texto =====
    _RE = re.compile(
        r'(?P<side>buy|sell|comprar|vender)\b'
        r'(?:.*?\b(lots?|lotes?)\s*[:=]?\s*(?P<lots>[0-9]*\.?[0-9]+))?'
        r'(?:.*?\b(entry|preço|preco|@)\s*[:=]?\s*(?P<entry>[0-9]*\.?[0-9]+))?'
        r'(?:.*?\bsl\b\s*[:=]?\s*(?P<sl>[0-9]*\.?[0-9]+))?'
        r'(?:.*?\btp\b\s*[:=]?\s*(?P<tp>[0-9]*\.?[0-9]+))?'
        r'(?:.*?\b(invert|inverter|invertido)\b\s*[:=]?\s*(?P<invert>[01]|true|false))?',
        flags=re.IGNORECASE
    )

    def parse_order(self, text: str) -> Optional[dict]:
        m = self._RE.search(text)
        if not m:
            return None
        side_raw = m.group("side").lower()
        side = "BUY" if side_raw in ("buy", "comprar") else "SELL"
        lots = float(m.group("lots")) if m.group("lots") else None
        entry = float(m.group("entry")) if m.group("entry") else None
        sl    = float(m.group("sl")) if m.group("sl") else None
        tp    = float(m.group("tp")) if m.group("tp") else None
        inv   = m.group("invert")
        invert_flag = 1 if (inv and inv.lower() in ("1","true")) else 0
        return {"side": side, "lots": lots, "entry": entry, "sl": sl, "tp": tp, "invert_flag": invert_flag}

    # ===== execução =====
    def _px_now(self, side_eff: str) -> float:
        import MetaTrader5 as mt5
        ti = mt5.symbol_info_tick(self.symbol)
        if not ti:
            raise RuntimeError("Tick vazio para o símbolo")
        if side_eff == "BUY" and getattr(ti, "ask", None):
            return float(ti.ask)
        if side_eff == "SELL" and getattr(ti, "bid", None):
            return float(ti.bid)
        return float((ti.ask + ti.bid) / 2.0)

    def order(self,
              side: str,
              lots: Optional[float] = None,
              entry: Optional[float] = None,
              sl: Optional[float] = None,
              tp: Optional[float] = None,
              invert_flag: Optional[int] = None,
              near_pct: Optional[float] = None,
              atr_now: Optional[float] = None) -> dict:

        inv = self.default_invert_flag if invert_flag is None else int(invert_flag)
        side_eff = ("SELL" if side.upper()=="BUY" else "BUY") if inv==1 else side.upper()
        px_now = self._px_now(side_eff)

        lots = lots if lots is not None else getattr(self.trader, "default_lots", 0.08)
        near = self.default_near_pct if near_pct is None else float(near_pct)

        _entry = px_now if entry is None else float(entry)
        _sl, _tp = sl, tp
        if (sl is None or tp is None) and self.sanitize_fn is not None:
            try:
                _, _sl_s, _tp_s = self.sanitize_fn(px_now, "COMPRA" if side_eff=="BUY" else "VENDA", None, atr_now or 1.0)
                _sl = _sl if _sl is not None else _sl_s
                _tp = _tp if _tp is not None else _tp_s
            except Exception:
                pass

        # chama sua função de envio (sem mexer nela)
        try:
            res = self.send_order_fn(
                self.trader, side_eff, px_now, _entry, _sl or 0.0, _tp or 0.0,
                lots=lots, near_pct=near
            )
            return {"ok": True, "result": str(res), "side": side_eff, "lots": lots, "entry": _entry, "sl": _sl, "tp": _tp}
        except Exception as e:
            return {"ok": False, "error": f"Falha ao enviar ordem: {e}"}

    def order_from_text(self, text: str, near_pct: Optional[float] = None) -> dict:
        spec = self.parse_order(text)
        if not spec:
            return {"ok": False, "error": "Não consegui entender a ordem no texto."}
        return self.order(**spec, near_pct=near_pct)
