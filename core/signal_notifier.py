# core/signal_notifier.py
from __future__ import annotations
import os
import time
import hashlib
import requests
from typing import Iterable, List, Tuple, Optional, Dict, Any


class SignalNotifier:
    """
    Envia alerta ao Telegram quando:
      - score >= min_score
      - TODOS os timeframes requeridos estão em COMPRA

    'usados' deve ser uma lista de tuplas: (tf, padrao, sinal, prob, peso).
    Ex.: [("M5","Hammer","COMPRA",80,1.0), ("M15","Hammer","COMPRA",80,1.5), ...]

    Anti-spam:
      - cooldown_sec: intervalo mínimo entre alertas
      - dedup por assinatura do conteúdo (não repete o mesmo relatório)
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        required_tfs: Optional[Iterable[str]] = None,
        min_score: float = 6.5,
        cooldown_sec: int = 180,
        timeout_sec: int = 12,
    ):
        self.token = token or os.getenv("TELEGRAM_TOKEN", "").strip()
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.required_tfs = [str(t).upper() for t in (required_tfs or [])]
        self.min_score = float(min_score)
        self.cooldown_sec = int(cooldown_sec)
        self.timeout_sec = int(timeout_sec)

        self._last_sent_at: float = 0.0
        self._last_signature: str = ""

    # -------------------- API pública --------------------

    def evaluate_and_notify(
        self,
        veredito_local: str,
        score: float,
        usados: List[Tuple[str, str, str, float, float]],
        min_prob_cfg: Optional[float] = None,
    ) -> bool:
        """
        Checa as condições e envia o mini-relatório.
        Retorna True se enviou, False caso contrário.
        """
        ver = (veredito_local or "").strip().upper()
        if ver != "COMPRA":
            return False
        if float(score) < self.min_score:
            return False

        ok, tf_map = self._all_required_buy(usados)
        if not ok:
            return False

        text = self._format_report(
            veredito_local=veredito_local,
            score=score,
            tf_map=tf_map,
            min_prob_cfg=min_prob_cfg,
        )
        return self._maybe_send(text)

    # -------------------- Internos --------------------

    def _all_required_buy(
        self,
        usados: List[Tuple[str, str, str, float, float]],
    ) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        """
        Retorna (ok, tf_map)

        ok == True se TODOS os required_tfs aparecem em COMPRA.
        tf_map: { "M5": {"padrao":..., "prob":..., "peso":...}, ...}
        Se houver mais de um item por TF, pegamos o de maior prob.
        """
        # Seleciona o melhor por TF (maior prob) somente se sinal == COMPRA
        best_by_tf: Dict[str, Dict[str, Any]] = {}
        for tf, padrao, sinal, prob, peso in usados:
            tfU = str(tf).upper()
            if sinal and str(sinal).upper() == "COMPRA":
                cur = best_by_tf.get(tfU)
                if (cur is None) or (float(prob) > float(cur["prob"])):
                    best_by_tf[tfU] = {"padrao": padrao, "prob": float(prob), "peso": float(peso)}

        # Se required_tfs não foi configurado: usa todos TFs que apareceram no 'usados'
        req = self.required_tfs or list(best_by_tf.keys())

        # Todos exigidos precisam estar presentes
        for tf in req:
            if tf not in best_by_tf:
                return False, best_by_tf

        return True, {tf: best_by_tf[tf] for tf in req}

    def _format_report(
        self,
        veredito_local: str,
        score: float,
        tf_map: Dict[str, Dict[str, Any]],
        min_prob_cfg: Optional[float] = None,
    ) -> str:
        """
        Gera o mini-relatório no formato solicitado.
        Mantém a ordem dos TFs conforme 'required_tfs' quando disponível.
        """
        header = f"Veredito Local (min_prob={int(min_prob_cfg) if min_prob_cfg is not None else '-' } ): {veredito_local} | score={score:.1f}"
        lines = [header]

        # Ordem dos TFs: prioriza required_tfs; se vazio, usa o dict na ordem natural
        order = self.required_tfs or list(tf_map.keys())
        for tf in order:
            d = tf_map.get(tf)
            if not d:
                continue
            padrao = d["padrao"]
            prob = int(round(d["prob"]))
            peso = d["peso"]
            lines.append(f" - {tf}: {padrao} | COMPRA | {prob}% (peso {peso})")

        return "\n".join(lines)

    def _maybe_send(self, text: str) -> bool:
        """
        Envia ao Telegram se:
          - token/chat_id presentes
          - cooldown respeitado
          - assinatura do conteúdo diferente da última enviada
        """
        if not self.token or not self.chat_id:
            # Sem credenciais → não envia, mas não quebra o fluxo
            print("[SignalNotifier] Telegram desabilitado (sem token/chat_id).")
            return False

        now = time.time()
        if now - self._last_sent_at < self.cooldown_sec:
            return False

        sig = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sig == self._last_signature:
            return False

        ok = self._send_telegram(text)
        if ok:
            self._last_signature = sig
            self._last_sent_at = now
        return ok

    def _send_telegram(self, text: str) -> bool:
        """
        Envia mensagem simples (texto puro) ao Telegram.
        """
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            r = requests.post(url, json=payload, timeout=self.timeout_sec)
            if r.status_code != 200:
                print("[SignalNotifier] Telegram erro:", r.text)
                return False
            return True
        except Exception as e:
            print("[SignalNotifier] Exceção ao enviar Telegram:", e)
            return False
