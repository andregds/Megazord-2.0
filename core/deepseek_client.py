"""DeepSeekClient
Cliente HTTP para a API da DeepSeek (Chat Completions).

- Endpoint oficial: POST https://api.deepseek.com/chat/completions
  (Também funciona com base_url https://api.deepseek.com/v1)
- Autenticação: Bearer <API_KEY>
- Modelos: deepseek-chat (não-thinking), deepseek-reasoner (thinking)

Uso:
    c = DeepSeekClient()
    texto = c.analisar("Seu resumo técnico aqui")
    print(texto)

Requer:
    - requests
    - config.py com DEEPSEEK_API_KEY (e opcionalmente DEEPSEEK_BASE_URL/DEEPSEEK_CHAT_PATH)
"""

from __future__ import annotations

import os
import time
import json
import requests
from typing import Optional, Dict, Any

# Tenta pegar da config; se não existir, usa env/defaults
try:
    from config import DEEPSEEK_API_KEY  # obrigatório
except Exception:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

try:
    from config import DEEPSEEK_BASE_URL  # opcional
except Exception:
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

try:
    from config import DEEPSEEK_CHAT_PATH  # opcional
except Exception:
    DEEPSEEK_CHAT_PATH = os.getenv("DEEPSEEK_CHAT_PATH", "/chat/completions")


class DeepSeekClient:
    """Encapsula chamadas à DeepSeek com timeout, retries e backoff exponencial leve."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        chat_path: Optional[str] = None,
        timeout: float = 20.0,
    ):
        self.api_key = (api_key or DEEPSEEK_API_KEY).strip()
        self.base_url = (base_url or DEEPSEEK_BASE_URL).rstrip("/")
        self.chat_path = (chat_path or DEEPSEEK_CHAT_PATH) or "/chat/completions"
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("DeepSeek API key não configurada (DEEPSEEK_API_KEY).")

    # ---- internos ----
    def _url(self) -> str:
        return f"{self.base_url}{self.chat_path}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ---- pública ----
    def analisar(
        self,
        resumo: str,
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        retries: int = 2,
        backoff: float = 4.0,
        timeout: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        """
        Envia o resumo técnico e devolve o texto do modelo.
        Retorna string começando com 'Erro DeepSeek:' em caso de falha (compatível com seu main).
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Você é um analista profissional de day trade em XAU/USD."},
                {"role": "user", "content": resumo},
            ],
            "temperature": float(temperature),
            "stream": bool(stream),
        }

        url = self._url()
        last_err = None
        to = timeout if timeout is not None else self.timeout

        for attempt in range(retries + 1):
            try:
                resp = requests.post(url, headers=self._headers(), json=payload, timeout=to)
                # Sucesso
                if resp.status_code == 200:
                    data = resp.json()
                    # formato compatível com OpenAI
                    return data["choices"][0]["message"]["content"].strip()

                # Erros temporários que valem retry
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}: {resp.text}"
                    if attempt < retries:
                        time.sleep(backoff * (attempt + 1))
                        continue
                # Outros: retorne mensagem clara
                return f"Erro DeepSeek: HTTP {resp.status_code} - {resp.text}"

            except requests.RequestException as e:
                last_err = str(e)
                if attempt < retries:
                    time.sleep(backoff * (attempt + 1))
                    continue

        return f"Erro DeepSeek: {last_err or 'falha desconhecida'}"


__all__ = ["DeepSeekClient"]


# Teste rápido opcional:
if __name__ == "__main__":
    try:
        c = DeepSeekClient()
        print("Endpoint:", c._url())
        print(c.analisar("Teste rápido: responda apenas 'OK'.", retries=0, timeout=10))
    except Exception as e:
        print("Self-test DeepSeekClient falhou:", e)
