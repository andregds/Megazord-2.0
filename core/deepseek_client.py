from __future__ import annotations
import os
import time
import json
import requests
from typing import Optional, Dict, Any

# Tenta pegar da config; se não existir, usa env/defaults
try:
    from config import DEEPSEEK_API_KEY  # obrigatório
except ImportError:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

try:
    from config import DEEPSEEK_BASE_URL  # opcional
except ImportError:
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

try:
    from config import DEEPSEEK_CHAT_PATH  # opcional
except ImportError:
    DEEPSEEK_CHAT_PATH = os.getenv("DEEPSEEK_CHAT_PATH", "/chat/completions")

class DeepSeekClient:
    """Encapsula chamadas à DeepSeek com timeout, retries e backoff exponencial leve."""
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        chat_path: Optional[str] = None,
        timeout: float = 60.0,
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

    def calcular_sl_tp(self, price_now: float) -> Tuple[Optional[float], Optional[float]]:
        # Exemplo de lógica para calcular SL e TP
        sl = price_now - 10  # Exemplo: SL 10 pontos abaixo do preço atual
        tp = price_now + 20  # Exemplo: TP 20 pontos acima do preço atual
        return sl, tp

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ---- pública ----
    def analisar(
        self,
        user_message: str,
        system_message: str = "Você é um assistente útil.",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        retries: int = 2,
        backoff: float = 4.0,

       #timeout: Optional[float] = None,
        timeout: Optional[float] = 60,
        stream: bool = False,
    ) -> str:
        """
        Envia uma mensagem ao modelo DeepSeek com uma mensagem de sistema configurável.
        Retorna a resposta do modelo como string.
        Retorna string começando com 'Erro DeepSeek:' em caso de falha.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
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

    def analisar_trade(
        self,
        resumo_tecnico: str,
        model: str = "deepseek-reasoner", # Usando deepseek-reasoner para decisões mais complexas
        temperature: float = 0.1, # Temperatura mais baixa para decisões mais assertivas
        retries: int = 3,
        backoff: float = 5.0,
        #timeout: Optional[float] = None,
        timeout: Optional[float] = 60,
        stream: bool = False,
    ) -> str:
        """
        Atua como um especialista institucional em operações day trade no mercado financeiro e forex XAU/USD.
        O robô precisa atuar de forma máxima assertiva nas suas decisões.
        Define Stop-Loss e Take-Profit automaticamente, gerenciando o risco.
        É o único tomador de decisões para ações de compra/venda, incluindo SL/TP.
        Valida que a decisão de compra ou venda esteja alinhada aos indicadores favoráveis.
        O log deve exibir de forma resumida e clara o motivo da decisão/veredito.
        Trata valores ausentes de SL e TP, evitando erros de conversão e garantindo robustez.

        Args:
            resumo_tecnico (str): O resumo técnico ou dados de mercado para análise.
            model (str): O modelo DeepSeek a ser usado (deepseek-reasoner para análise mais profunda).
            temperature (float): A temperatura para a geração do modelo (mais baixa para assertividade).
            retries (int): Número de tentativas em caso de falha.
            backoff (float): Tempo de espera entre as tentativas.
            timeout (Optional[float]): Tempo limite para a requisição.
            stream (bool): Se a resposta deve ser transmitida.

        Returns:
            str: A decisão de trade, incluindo SL/TP e o motivo da decisão.
        """
        system_prompt = (
            "Você é um especialista institucional em operações de day trade e forex, focado especificamente em XAU/USD. "
            "Sua função é tomar decisões de compra/venda com a máxima assertividade, como um player institucional. "
            "Para cada decisão, você DEVE definir um Stop-Loss (SL) e um Take-Profit (TP) de forma automática e inteligente, "
            "gerenciando o risco de forma proativa. Você é o ÚNICO tomador de decisões para as ações de compra/venda, "
            "incluindo as configurações de SL e TP. Sua análise deve validar que a decisão de compra ou venda "
            "esteja sempre alinhada aos indicadores técnicos e fundamentais mais favoráveis. "
            "Ao apresentar sua decisão, forneça um VEREDITO claro e conciso (COMPRAR, VENDER, MANTER) "
            "e, em seguida, detalhe o MOTIVO da decisão, os valores de STOP-LOSS e TAKE-PROFIT. "
            "Certifique-se de que, se SL ou TP não puderem ser determinados por algum motivo, você deve indicar isso "
            "claramente e não retornar valores inválidos ou ausentes que possam causar erros de conversão. "
            "Sua resposta deve ser estruturada para fácil parseamento, idealmente em um formato como:\n\n"
            "VEREDITO: [COMPRAR/VENDER/MANTER]\n"
            "MOTIVO: [Explicação detalhada da decisão baseada em indicadores e análise institucional]\n"
            "STOP-LOSS: [Valor ou 'Não Definido']\n"
            "TAKE-PROFIT: [Valor ou 'Não Definido']\n\n"
            "Considere sempre o contexto de day trade, buscando oportunidades de curto prazo com alta probabilidade."
        )
        return self.analisar(
            user_message=resumo_tecnico,
            system_message=system_prompt,
            model=model,
            temperature=temperature,
            retries=retries,
            backoff=backoff,
            timeout=timeout,
            stream=stream,
        )




__all__ = ["DeepSeekClient"]

# Teste rápido opcional:
if __name__ == "__main__":
    try:
        c = DeepSeekClient()
        print("Endpoint:", c._url())

        # Teste do método analisar_trade com um cenário hipotético
        print("\n--- Teste de Análise de Trade ---")
        exemplo_resumo = (
            "Análise técnica para XAU/USD em M15: RSI em 75 (sobrecomprado), "
            "MACD cruzando para baixo, preço testando resistência em 2050.00. "
            "Volume de venda aumentando. Notícias recentes indicam fortalecimento do USD."
        )
        trade_decision = c.analisar_trade(exemplo_resumo)
        print(trade_decision)

        # Teste do método analisar com prompt padrão
        print("\n--- Teste de Análise Geral ---")
        print(c.analisar("Teste rápido: responda apenas 'OK'.", retries=0, timeout=30))

    except Exception as e:
        print("Self-test DeepSeekClient falhou:", e)

