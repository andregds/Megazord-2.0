import requests
import logging
from typing import Dict, Any

logger = logging.getLogger("Megazord.ollama")


def analisar_trade_local(resumo_indicadores: str, timeout: float = 4.5) -> Dict[str, Any]:
    """Enviar resumo de indicadores ao Ollama local e retornar JSON compacto.

    Regras importantes (conforme solicitado):
    - URL termina em /api/chat (SEM barra final)
    - Método: requests.post
    - Payload: estrutura `messages` + `format: "json"` + `options.num_predict = 45`
    - Timeout padrão: 4.5 segundos

    Retorna: dict com resposta JSON do modelo em caso de 200, ou fallback com chave 'decisao':'AGUARDAR'.
    """
    url = "http://109.199.107.136:11434/api/chat"

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "qwen2.5:1.5b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é o validador de risco do Robô Megazord para XAU/USD. "
                    "Responda EXCLUSIVAMENTE um objeto JSON válido. O motivo deve ter no máximo 8 palavras. "
                    "Esquema: {\"decisao\": \"COMPRA\"|\"VENDA\"|\"AGUARDAR\", \"sl\": float, \"tp\": float, \"motivo\": \"string\"}."
                ),
            },
            {"role": "user", "content": resumo_indicadores},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 45},
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                logger.error("Resposta não-JSON do Ollama: %s", response.text)
                return {"decisao": "AGUARDAR", "motivo": "Resposta inválida do Ollama"}
        else:
            logger.error("❌ Erro na API Ollama: Status %s - %s", response.status_code, response.text)
            return {"decisao": "AGUARDAR", "motivo": f"Erro HTTP {response.status_code}"}

    except requests.exceptions.Timeout:
        logger.warning("⏳ Timeout atingido ao conectar com Ollama. Retornando fallback seguro.")
        return {"decisao": "AGUARDAR", "motivo": "Ollama Timeout"}
    except requests.exceptions.RequestException as e:
        logger.error("Erro na requisição Ollama: %s", e)
        return {"decisao": "AGUARDAR", "motivo": "Erro de comunicação"}


__all__ = ["analisar_trade_local"]

