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
    from config import OLLAMA_API_BASE_URL, OLLAMA_MODEL_NAME
except ImportError:
    OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", None)
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", None)

# Para compatibilidade antiga deixamos variáveis vazias para DeepSeek (não usadas)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "")
DEEPSEEK_CHAT_PATH = os.getenv("DEEPSEEK_CHAT_PATH", "")

class DeepSeekClient:
    """Encapsula chamadas à DeepSeek com timeout, retries e backoff exponencial leve."""
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        chat_path: Optional[str] = None,
        # Aumenta o timeout padrão para requisições ao Ollama local (antes 4.5s)
        timeout: float = 45.0,
    ):
        # Quando integrado ao Ollama local, a API key não é obrigatória
        self.api_key = (api_key or "").strip()
        # Se houver OLLAMA_API_BASE_URL configurado, o cliente usa Ollama
        self.ollama_base = (OLLAMA_API_BASE_URL or None)
        self.ollama_model = (OLLAMA_MODEL_NAME or None)
        if self.ollama_base:
            self.base_url = self.ollama_base.rstrip("/")
            self.chat_path = None
        else:
            # fallback (compatibilidade)
            self.base_url = (base_url or DEEPSEEK_BASE_URL.rstrip("/")) if base_url or DEEPSEEK_BASE_URL else ""
            self.chat_path = (chat_path or DEEPSEEK_CHAT_PATH) or "/chat/completions"
        self.timeout = float(timeout)

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

    def _emit_final_veredito(self, decision: str, sl: float, tp: float, source: str) -> Dict[str, Any]:
        """Build a canonical final JSON for the trade verdict, log it to Logs/log_api.log and return it.

        The returned dict has keys: decision, sl, tp, source, ts
        """
        final = {
            "decision": decision,
            "sl": float(sl) if sl is not None else 0.0,
            "tp": float(tp) if tp is not None else 0.0,
            "source": source,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            log_dir = os.path.join(os.getcwd(), "Logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "log_api.log")
            entry = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " | FINAL_VEREDITO: " + json.dumps(final, ensure_ascii=False) + "\n"
            self._rotate_and_append(log_path, entry)
        except Exception:
            logger.debug("Falha ao gravar final veredito no log")
        return final

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
                # Se Ollama estiver configurado, envie um payload compacto em JSON
                # solicitando explicitamente resposta em JSON para evitar textos longos.
                if self.ollama_base:
                    # Envia mensagens no formato esperado e força 'format':'json'
                    payload: Dict[str, Any] = {
                        "model": (self.ollama_model or model),
                        "messages": messages,
                        "format": "json",
                        "stream": False,
                        "options": {
                            "temperature": float(temperature),
                            # limite de tokens gerados para evitar respostas longas
                            "num_predict": 45,
                        },
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

                try:
                    resp = requests.post(url, headers=self._headers(), json=payload, timeout=to)
                    resp.raise_for_status()
                    data = resp.json()
                except requests.exceptions.Timeout as e:
                    logger.warning(f"Timeout ao conectar com o modelo (url={url}) usando timeout={to}s: {e}")
                    # Tentativa adicional com timeout triplicado (warm/retry)
                    try:
                        to2 = float(to) * 3
                        logger.info(f"Tentando nova requisição com timeout triplicado: {to2}s")
                        resp2 = requests.post(url, headers=self._headers(), json=payload, timeout=to2)
                        resp2.raise_for_status()
                        data = resp2.json()
                        # atualiza variável resp para efeitos de logging abaixo
                        resp = resp2
                    except requests.exceptions.Timeout as e2:
                        logger.error(f"Timeout (retry) ao conectar com o modelo (url={url}) usando timeout={to2}s: {e2}")
                        return {"error": "Timeout: Ollama request timed out (retry)", "fallback": {"decisao": "AGUARDAR", "motivo": "Erro de comunicação com Ollama local (timeout)"}}
                    except requests.exceptions.ConnectionError as e2:
                        logger.error(f"ConnectionError na retry ao conectar com o modelo (url={url}): {e2}")
                        return {"error": "ConnectionError: failed to connect to Ollama on retry", "fallback": {"decisao": "AGUARDAR", "motivo": "Erro de comunicação com Ollama local (connection)"}}
                    except json.JSONDecodeError as e2:
                        logger.error(f"JSON decode error na resposta do modelo (retry url={url}): {e2}")
                        return {"error": "JSONDecodeError: invalid JSON from Ollama (retry)", "fallback": {"decisao": "AGUARDAR", "motivo": "Resposta inválida do Ollama local"}}
                    except Exception as e2:
                        logger.error(f"Erro inesperado na retry ao modelo (url={url}): {e2}")
                        return {"error": f"Unexpected error (retry): {e2}", "fallback": {"decisao": "AGUARDAR", "motivo": "Erro de comunicação com Ollama local"}}
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"ConnectionError ao conectar com o modelo (url={url}): {e}")
                    return {"error": "ConnectionError: failed to connect to Ollama", "fallback": {"decisao": "AGUARDAR", "motivo": "Erro de comunicação com Ollama local (connection)"}}
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error na resposta do modelo (url={url}): {e}")
                    return {"error": "JSONDecodeError: invalid JSON from Ollama", "fallback": {"decisao": "AGUARDAR", "motivo": "Resposta inválida do Ollama local"}}
                except Exception as e:
                    logger.error(f"Erro inesperado na request ao modelo (url={url}): {e}")
                    return {"error": f"Unexpected error: {e}", "fallback": {"decisao": "AGUARDAR", "motivo": "Erro de comunicação com Ollama local"}}
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

                # Tentar extrair resposta JSON do Ollama (ou conteúdo textual que contenha JSON)
                extracted = None
                parsed_json = None
                try:
                    # Se a resposta já for um dict com campo 'output' (Ollama), tente extrair
                    if isinstance(data, dict):
                        # Caso comum: 'output' é uma lista de strings ou itens
                        if "output" in data:
                            out = data["output"]
                            if isinstance(out, list) and out:
                                # tentar primeiro se o primeiro elemento já for dict/json
                                first = out[0]
                                if isinstance(first, dict):
                                    parsed_json = first
                                else:
                                    # tentar carregar como JSON de string
                                    try:
                                        parsed_json = json.loads(str(first))
                                    except Exception:
                                        extracted = "\n".join(str(x) for x in out)
                        # outras chaves úteis
                        if parsed_json is None and "response" in data:
                            r = data["response"]
                            if isinstance(r, dict):
                                parsed_json = r
                            else:
                                try:
                                    parsed_json = json.loads(str(r))
                                except Exception:
                                    extracted = str(r)
                        # campo 'result' também pode conter a carga útil
                        if parsed_json is None and "result" in data:
                            res = data["result"]
                            if isinstance(res, dict):
                                # procurar por subcampos
                                for k in ("output", "text", "content", "response", "generated"):
                                    if k in res:
                                        val = res[k]
                                        if isinstance(val, dict):
                                            parsed_json = val
                                            break
                                        if isinstance(val, list):
                                            try:
                                                parsed_json = json.loads(str(val[0]))
                                            except Exception:
                                                extracted = "\n".join(str(x) for x in val)
                                                break
                                        else:
                                            try:
                                                parsed_json = json.loads(str(val))
                                            except Exception:
                                                extracted = str(val)
                                                break
                        # OpenAI-like fallback
                        if parsed_json is None and "choices" in data and data.get("choices"):
                            c = data["choices"][0]
                            if isinstance(c, dict):
                                if "message" in c and isinstance(c["message"], dict):
                                    extracted = c["message"].get("content")
                                elif "text" in c:
                                    extracted = c.get("text")
                    # Se ainda não temos parsed_json, e extrated for vazio, tentar serializar todo 'data'
                    if parsed_json is None and extracted is None:
                        # tenta transformar em string e buscar JSON dentro
                        s = None
                        try:
                            s = json.dumps(data, ensure_ascii=False)
                        except Exception:
                            s = str(data)
                        # procurar por um objeto JSON dentro da string
                        try:
                            parsed_json = json.loads(s)
                        except Exception:
                            # tentar extrair texto puro
                            extracted = s
                except Exception:
                    parsed_json = None
                    # em erro, mantenha extracted como None para fallback abaixo

                # Se encontramos JSON já parseado, retorne-o juntamente com a string
                if parsed_json is not None:
                    try:
                        # também gerar uma versão 'content' para compatibilidade com quem espera texto
                        content_text = json.dumps(parsed_json, ensure_ascii=False)
                    except Exception:
                        content_text = str(parsed_json)
                    return {"content": content_text, "json": parsed_json}

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
        # Se True, instruir o modelo a PROIBIR a opção 'AGUARDAR' e escolher COMPRA ou VENDA
        decision_required: bool = False,
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
        # Prompt mais enxuto e forçando JSON quando possível — reduz tokens e evita divagações.
        # Se `decision_required` for True, proibimos explicitamente 'AGUARDAR'.
        decision_rule = "Você NÃO pode usar 'AGUARDAR' — escolha apenas COMPRA ou VENDA." if decision_required else "Você pode escolher COMPRA, VENDA ou AGUARDAR."
        system_prompt = (
            "Você é um validador de risco institucional para XAU/USD. Responda EXCLUSIVAMENTE um objeto JSON válido e NADA mais."
            " O motivo deve ter no máximo 8 palavras. "
            f"{decision_rule} "
            "Esquema JSON obrigatório: {\"decisao\": \"COMPRA\"|\"VENDA\", \"sl\": float, \"tp\": float, \"motivo\": \"string\"}."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": resumo_tecnico},
        ]

        response_obj = self._call_api(messages, model, temperature, retries, backoff, timeout, stream)

        # defaults
        decision = "AGUARDAR"
        sl = 0.0
        tp = 0.0
        reason = "Não foi possível extrair o motivo da decisão."
        response_text = response_obj.get("content", response_obj.get("error", "Erro DeepSeek: resposta vazia ou erro desconhecido."))

        # If API returned an error, keep fallback
        if "error" in response_obj:
            logger.error("Erro na chamada da API DeepSeek: %s", response_obj.get('error'))
            final = self._emit_final_veredito("AGUARDAR", 0.0, 0.0, (self.ollama_model or model))
            return {
                "veredito_text": response_text,
                "decision": "AGUARDAR",
                "sl": 0.0,
                "tp": 0.0,
                "reason": response_obj.get('error'),
                "final": final,
            }

        # If Ollama returned structured JSON (we request format:json), use it directly.
        if "json" in response_obj and isinstance(response_obj["json"], dict):
            j = response_obj["json"]
            # Normalize keys (Portuguese/English)
            def _get_key(obj, *keys):
                for k in keys:
                    if k in obj:
                        return obj[k]
                return None

            raw_dec = _get_key(j, "decisao", "decisão", "decision")
            if raw_dec is not None:
                rd = str(raw_dec).strip().upper()
                if rd in ("COMPRA", "BUY"):
                    decision = "COMPRA"
                elif rd in ("VENDA", "SELL"):
                    decision = "VENDA"
                elif rd in ("AGUARDAR", "HOLD", "WAIT"):
                    decision = "AGUARDAR"

            # SL/TP keys
            sl_val = _get_key(j, "sl", "SL", "stop_loss")
            tp_val = _get_key(j, "tp", "TP", "take_profit")
            try:
                if sl_val is not None:
                    sl = float(sl_val)
            except Exception:
                sl = 0.0
            try:
                if tp_val is not None:
                    tp = float(tp_val)
            except Exception:
                tp = 0.0

            motivo_val = _get_key(j, "motivo", "reason", "motivation")
            if motivo_val is not None:
                try:
                    reason = str(motivo_val).strip()
                except Exception:
                    reason = ""

            # ensure AGUARDAR implies zero sl/tp
            if decision == "AGUARDAR":
                sl = 0.0
                tp = 0.0

            # build a compact veredito_text from JSON for logs
            try:
                response_text = json.dumps(response_obj["json"], ensure_ascii=False)
            except Exception:
                response_text = str(response_obj["json"])

            final = self._emit_final_veredito(decision, sl, tp, (self.ollama_model or model))
            return {
                "veredito_text": response_text,
                "decision": decision,
                "sl": sl,
                "tp": tp,
                "reason": reason,
                "final": final,
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


        final = self._emit_final_veredito(decision, sl, tp, (self.ollama_model or model))
        return {
            "veredito_text": response_text, # Texto completo da resposta do DeepSeek
            "decision": decision,
            "sl": sl,
            "tp": tp,
            "reason": reason, # O motivo da decisão
            "final": final,
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

