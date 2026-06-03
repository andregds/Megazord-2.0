from __future__ import annotations
import os
import time
import json
import requests
import re # Importar o módulo de expressões regulares
import logging # Importar o módulo de logging
from typing import Optional, Dict, Any, Tuple

# Configura o logger da aplicação (se já não estiver configurado em main.py)
# É uma boa prática ter o logger configurado globalmente ou passá-lo
logger = logging.getLogger("Megazord")
if not logger.handlers: # Configura apenas se não houver handlers, para evitar duplicação
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("Megazord")


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
# Tentar importar configuração do Ollama (opcional)
try:
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME
except ImportError:
    OLLAMA_BASE_URL = None
    OLLAMA_MODEL_NAME = None

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
        # Se OLLAMA estiver configurado, preferimos ele
        self.ollama_base = (OLLAMA_BASE_URL or None)
        self.ollama_model = (OLLAMA_MODEL_NAME or None)
        if self.ollama_base:
            self.base_url = self.ollama_base.rstrip("/")
            self.chat_path = None
        else:
            self.base_url = (base_url or DEEPSEEK_BASE_URL).rstrip("/")
            self.chat_path = (chat_path or DEEPSEEK_CHAT_PATH) or "/chat/completions"
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("DeepSeek API key não configurada (DEEPSEEK_API_KEY).")

    # ---- internos ----
    def _url(self) -> str:
        if self.ollama_base:
            return self.base_url
        return f"{self.base_url}{self.chat_path}"

    def _rotate_and_append(self, path: str, text: str, max_bytes: int = 10 * 1024 * 1024, backups: int = 4) -> None:
        """Append text to path, rotating when size exceeds max_bytes."""
        try:
            if os.path.exists(path) and os.path.getsize(path) > max_bytes:
                for i in range(backups - 1, 0, -1):
                    s = f"{path}.{i}"
                    d = f"{path}.{i+1}"
                    if os.path.exists(s):
                        try:
                            os.replace(s, d)
                        except Exception:
                            pass
                try:
                    os.replace(path, f"{path}.1")
                except Exception:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text)
        except Exception:
            logger.debug("Não foi possível escrever no log_api.log")

    # Este método calcular_sl_tp não será mais usado diretamente para a decisão da IA,
    # pois a IA definirá SL/TP. Mantê-lo ou removê-lo depende se ele tem outro uso.
    # Por enquanto, vamos mantê-lo, mas ele não será chamado pelo main.py para a decisão da IA.
    def calcular_sl_tp(self, price_now: float) -> Tuple[Optional[float], Optional[float]]:
        # Exemplo de lógica para calcular SL e TP
        sl = price_now - 10  # Exemplo: SL 10 pontos abaixo do preço atual
        tp = price_now + 20  # Exemplo: TP 20 pontos acima do preço atual
        return sl, tp

    def _headers(self) -> Dict[str, str]:
        # Ollama local normalmente não exige Authorization header.
        h = {"Content-Type": "application/json"}
        if not self.ollama_base and self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    # ---- pública ----
    # O método 'analisar' original será adaptado para ser mais genérico e usado internamente
    # pelo 'analisar_trade' para fazer a chamada à API e parsear a resposta.
    def _call_api(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        retries: int,
        backoff: float,
        timeout: Optional[float],
        stream: bool,
    ) -> dict: # Retorna um dicionário com a resposta bruta ou erro
        url = self._url()
        to = timeout if timeout is not None else self.timeout

        for attempt in range(retries + 1):
            try:
                # Se Ollama estiver configurado, convertemos as mensagens para um prompt único
                if self.ollama_base:
                    # Junta mensagens system/user/assistant em um único prompt
                    parts = []
                    for m in messages:
                        role = m.get("role", "user").upper()
                        parts.append(f"[{role}] {m.get('content','')}")
                    prompt_text = "\n\n".join(parts)
                    payload: Dict[str, Any] = {
                        "model": (self.ollama_model or model),
                        "input": prompt_text,
                    }
                else:
                    payload: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": float(temperature),
                        "stream": bool(stream),
                    }

                # prepare logging
                try:
                    log_dir = os.path.join(os.getcwd(), "Logs")
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, "log_api.log")
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    pre_header = f"{ts} | DeepSeek REQUEST -> model={model} url={url} timeout={to}s attempt={attempt+1}/{retries+1}\n"
                    # mask headers
                    try:
                        headers_masked = dict(self._headers())
                        if "Authorization" in headers_masked:
                            headers_masked["Authorization"] = "***MASKED***"
                    except Exception:
                        headers_masked = {"Authorization": "***MASKED***"}

                    log_text = pre_header
                    try:
                        log_text += "REQUEST_HEADERS:\n" + json.dumps(headers_masked, ensure_ascii=False, default=str) + "\n"
                    except Exception:
                        log_text += "REQUEST_HEADERS: <unserializable>\n"
                    try:
                        log_text += "REQUEST_PAYLOAD:\n" + json.dumps(payload, ensure_ascii=False, default=str) + "\n"
                    except Exception:
                        log_text += "REQUEST_PAYLOAD: <unserializable>\n"
                    print(pre_header.strip())
                    logger.info(pre_header.strip())
                    self._rotate_and_append(log_path, log_text)
                except Exception:
                    logger.debug("Falha ao preparar log de requisição DeepSeek")

                resp = requests.post(url, headers=self._headers(), json=payload, timeout=to)
                resp.raise_for_status() # Levanta um HTTPError para códigos de status de erro (4xx ou 5xx)

                data = resp.json()
                if stream:
                    # Se stream for True, a lógica de retorno é diferente.
                    # Por simplicidade, para este caso, vamos assumir stream=False.
                    # Se precisar de stream, esta parte precisará de uma implementação mais complexa.
                    logger.warning("Streaming não implementado para _call_api neste contexto.")
                    return {"error": "Streaming not implemented for _call_api"}

                # log response
                try:
                    ts2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    post_header = f"{ts2} | Model RESPONSE <- status={resp.status_code} model={model}\n"
                    print(post_header.strip())
                    logger.info(post_header.strip())
                    try:
                        resp_text = post_header + "RESPONSE_BODY:\n" + json.dumps(data, ensure_ascii=False, default=str) + "\n"
                    except Exception:
                        resp_text = post_header + "RESPONSE_BODY: <unserializable>\n"
                    try:
                        self._rotate_and_append(log_path, resp_text)
                    except Exception:
                        pass
                except Exception:
                    logger.debug("Falha ao gravar log de resposta do modelo")

                # Tentar extrair texto da resposta de forma robusta (suporta Ollama e formato estilo OpenAI)
                extracted = None
                try:
                    # OpenAI-like
                    if isinstance(data, dict) and "choices" in data and data["choices"]:
                        c = data["choices"][0]
                        if isinstance(c, dict) and "message" in c and isinstance(c["message"], dict):
                            extracted = c["message"].get("content")
                        elif isinstance(c, dict) and "text" in c:
                            extracted = c.get("text")
                    # Ollama-like: possible keys 'result', 'output', 'response', 'generated'
                    if extracted is None and isinstance(data, dict):
                        if "result" in data and isinstance(data["result"], dict):
                            # buscar campos comuns
                            for k in ("output", "text", "content", "response"):
                                if k in data["result"]:
                                    val = data["result"][k]
                                    if isinstance(val, list):
                                        extracted = "\n".join(str(x) for x in val)
                                    else:
                                        extracted = str(val)
                                    break
                        if extracted is None and "output" in data:
                            out = data["output"]
                            if isinstance(out, list):
                                extracted = "\n".join(str(x) for x in out)
                            else:
                                extracted = str(out)
                        if extracted is None and "response" in data:
                            extracted = str(data["response"])
                        if extracted is None and "generated" in data:
                            extracted = str(data["generated"])                
                except Exception:
                    extracted = None

                if extracted:
                    return {"content": extracted.strip()}
                # último recurso: devolver erro com dump
                return {"error": f"No textual content found in model response", "raw": data}

            except requests.exceptions.RequestException as e:
                logger.error(f"Erro na requisição DeepSeek (tentativa {attempt+1}/{retries+1}): {e}")
                if attempt < retries:
                    time.sleep(backoff * (attempt + 1)) # Backoff exponencial
                else:
                    return {"error": f"Erro DeepSeek: {e}"}
            except json.JSONDecodeError as e:
                logger.error(f"Erro ao decodificar JSON da resposta DeepSeek (tentativa {attempt+1}/{retries+1}): {e}")
                if attempt < retries:
                    time.sleep(backoff * (attempt + 1))
                else:
                    return {"error": f"Erro DeepSeek: Resposta JSON inválida - {e}"}
            except Exception as e:
                logger.error(f"Erro inesperado na chamada DeepSeek (tentativa {attempt+1}/{retries+1}): {e}")
                if attempt < retries:
                    time.sleep(backoff * (attempt + 1))
                else:
                    return {"error": f"Erro DeepSeek: {e}"}

    def analisar( # Renomeado para 'analisar_generico' ou similar se o 'analisar_trade' for o principal
        self,
        user_message: str,
        system_message: str = "Você é um assistente útil.",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        retries: int = 2,
        backoff: float = 4.0,
        timeout: Optional[float] = 60,
        stream: bool = False,
    ) -> str:
        """
        Envia uma mensagem ao modelo DeepSeek com uma mensagem de sistema configurável.
        Retorna a resposta do modelo como string.
        Retorna string começando com 'Erro DeepSeek:' em caso de falha.
        Este método é mais genérico e não faz parsing estruturado.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response_obj = self._call_api(messages, model, temperature, retries, backoff, timeout, stream)
        if "content" in response_obj:
            return response_obj["content"]
        return response_obj.get("error", "Erro DeepSeek: falha desconhecida")


    def analisar_trade(
        self,
        resumo_tecnico: str,
        model: str = "deepseek-reasoner", # Usando deepseek-reasoner para decisões mais complexas
        temperature: float = 0.1, # Temperatura mais baixa para decisões mais assertivas
        retries: int = 3,
        backoff: float = 5.0,
        timeout: Optional[float] = 60,
        stream: bool = False,
    ) -> Dict[str, Any]: # O retorno agora é um dicionário
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
            Dict[str, Any]: Um dicionário contendo a decisão de trade, SL/TP e o texto completo da resposta.
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
            "Sua resposta DEVE ser estruturada para fácil parseamento, utilizando o seguinte formato:\n\n"
            "DECISÃO: [COMPRA/VENDA/AGUARDAR]\n" # Alterado para o formato que o main.py espera
            "SL: [Valor ou '0.0' se não aplicável/determinado]\n" # Alterado para SL: [Valor]
            "TP: [Valor ou '0.0' se não aplicável/determinado]\n" # Alterado para TP: [Valor]
            "MOTIVO: [Explicação detalhada da decisão baseada em indicadores e análise institucional]\n\n"
            "Considere sempre o contexto de day trade, buscando oportunidades de curto prazo com alta probabilidade."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": resumo_tecnico},
        ]

        response_obj = self._call_api(messages, model, temperature, retries, backoff, timeout, stream)

        decision = "AGUARDAR"
        sl = 0.0
        tp = 0.0
        reason = "Não foi possível extrair o motivo da decisão."
        response_text = response_obj.get("content", response_obj.get("error", "Erro DeepSeek: resposta vazia ou erro desconhecido."))

        if "error" in response_obj:
            logger.error(f"Erro na chamada da API DeepSeek: {response_obj['error']}")
            return {
                "veredito_text": response_text,
                "decision": "AGUARDAR",
                "sl": 0.0,
                "tp": 0.0,
                "reason": response_obj['error']
            }

        # --- INÍCIO DO PARSING DA RESPOSTA ESTRUTURADA ---
        # Regex para encontrar a decisão, SL, TP e o motivo
        # O padrão agora espera "DECISÃO:", "SL:", "TP:" e "MOTIVO:"
        decision_match = re.search(
            r"DECISÃO:\s*(COMPRA|VENDA|AGUARDAR)\s*\n"
            r"SL:\s*(\d+\.?\d*|Não Aplicável|0\.0)\s*\n" # Captura SL
            r"TP:\s*(\d+\.?\d*|Não Aplicável|0\.0)\s*\n" # Captura TP
            r"MOTIVO:\s*(.*)", # Captura o motivo, que pode ser multi-linha
            response_text,
            re.IGNORECASE | re.DOTALL # re.DOTALL para que '.' inclua quebras de linha no motivo
        )

        if decision_match:
            decision = decision_match.group(1).upper()

            # Extrair SL
            sl_str = decision_match.group(2)
            try:
                sl = float(sl_str)
            except ValueError:
                logger.warning(f"DeepSeek retornou SL inválido ou 'Não Aplicável': {sl_str}. Usando 0.0.")
                sl = 0.0

            # Extrair TP
            tp_str = decision_match.group(3)
            try:
                tp = float(tp_str)
            except ValueError:
                logger.warning(f"DeepSeek retornou TP inválido ou 'Não Aplicável': {tp_str}. Usando 0.0.")
                tp = 0.0

            reason = decision_match.group(4).strip()

            # Se a decisão for AGUARDAR, SL e TP devem ser 0.0
            if decision == "AGUARDAR":
                sl = 0.0
                tp = 0.0

        else:
            # Fallback se o DeepSeek não seguir o formato exato.
            # Esta lógica é mais robusta para garantir que uma decisão seja tomada,
            # mas o ideal é que a IA siga o formato.
            logger.warning("DeepSeek não seguiu o formato de decisão estruturado. Tentando inferir.")
            if "COMPRA" in response_text.upper() and "AGUARDAR" not in response_text.upper():
                decision = "COMPRA"
            elif "VENDA" in response_text.upper() and "AGUARDAR" not in response_text.upper():
                decision = "VENDA"
            else:
                decision = "AGUARDAR"

            # Tentar extrair SL/TP mesmo sem o formato explícito de decisão
            sl_match = re.search(r"(?:Stop-Loss|SL)\s*[:=]\s*(\d+\.?\d*)", response_text, re.IGNORECASE)
            tp_match = re.search(r"(?:Take-Profit|TP)\s*[:=]\s*(\d+\.?\d*)", response_text, re.IGNORECASE)
            if sl_match:
                try: sl = float(sl_match.group(1))
                except ValueError: logger.warning(f"Fallback: SL inválido {sl_match.group(1)}. Usando 0.0.")
            if tp_match:
                try: tp = float(tp_match.group(1))
                except ValueError: logger.warning(f"Fallback: TP inválido {tp_match.group(1)}. Usando 0.0.")

            # Se a decisão inferida for AGUARDAR, SL e TP devem ser 0.0
            if decision == "AGUARDAR":
                sl = 0.0
                tp = 0.0

            # Tentar extrair um motivo genérico se o formato não foi seguido
            reason = "Decisão inferida. Formato DeepSeek não seguido. Resposta original: " + response_text[:200] + "..."


        return {
            "veredito_text": response_text, # Texto completo da resposta do DeepSeek
            "decision": decision,
            "sl": sl,
            "tp": tp,
            "reason": reason # O motivo da decisão
        }
        # --- FIM DO PARSING DA RESPOSTA ESTRUTURADA ---

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
            "\n\nInstruções: DECISÃO: [COMPRA/VENDA/AGUARDAR]\nSL: [Valor]\nTP: [Valor]\nMOTIVO: [Explicação]"
        )
        trade_decision = c.analisar_trade(exemplo_resumo)
        print(f"Decisão: {trade_decision['decision']}")
        print(f"SL: {trade_decision['sl']}")
        print(f"TP: {trade_decision['tp']}")
        print(f"Motivo: {trade_decision['reason']}")
        print(f"Texto completo: {trade_decision['veredito_text']}")

        print("\n--- Teste de Análise de Trade (AGUARDAR) ---")
        exemplo_resumo_aguardar = (
            "Análise técnica para XAU/USD: Sinais conflitantes em múltiplos timeframes. "
            "RSI neutro, preço em consolidação. Nenhuma oportunidade clara. "
            "\n\nInstruções: DECISÃO: [COMPRA/VENDA/AGUARDAR]\nSL: [Valor]\nTP: [Valor]\nMOTIVO: [Explicação]"
        )
        trade_decision_aguardar = c.analisar_trade(exemplo_resumo_aguardar)
        print(f"Decisão: {trade_decision_aguardar['decision']}")
        print(f"SL: {trade_decision_aguardar['sl']}")
        print(f"TP: {trade_decision_aguardar['tp']}")
        print(f"Motivo: {trade_decision_aguardar['reason']}")
        print(f"Texto completo: {trade_decision_aguardar['veredito_text']}")

        print("\n--- Teste de Análise Geral (método analisar) ---")
        print(c.analisar("Teste rápido: responda apenas 'OK'.", retries=0, timeout=30))
    except Exception as e:
        print("Self-test DeepSeekClient falhou:", e)

