import json
import logging
from typing import Callable, Optional, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from app.commons.services.miscelaneous import load_prompts_generales


def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _extract_json(text: str) -> Tuple[Optional[Any], Optional[Exception]]:
    if text is None:
        return None, ValueError("Empty response")

    try:
        return json.loads(text), None
    except Exception:
        pass

    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped), None
    except Exception:
        pass

    try:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start:end + 1]
            return json.loads(candidate), None
    except Exception as e:
        return None, e

    return None, ValueError("Could not parse JSON after repairs")


def evaluar_circunstancias_marcus(
        llm: object,
        contexto_marcus: str,
        json_visual: str,
        json_transcripcion: str,
        *,
        schema_validator: Optional[Callable[[Any], Any]] = None,
        schema_description: Optional[str] = None,
        force_json_only: bool = True,
        max_retries: int = 1,
) -> Any:
    """
    Envía la información consolidada al LLM y garantiza que la salida sea JSON válido.
    Devuelve SIEMPRE un dict (JSON parseado). En caso de error devuelve {"error": "..."}.
    """
    try:
        base_prompt = load_prompts_generales("evaluar_circunstancias_marcus")
        if not base_prompt:
            return {"error": "❌ Prompt 'evaluar_circunstancias_marcus' no encontrado en YAML."}

        json_rules = (
            "You MUST respond with ONE valid JSON object only. "
            "Do not include any prose, prefixes, suffixes, markdown, or code fences. "
            "The response MUST be strictly parseable with JSON.parse / json.loads. "
            "Use double quotes for all keys and string values. No trailing commas."
        )
        if schema_description:
            json_rules += f" The JSON MUST conform to this structure: {schema_description}"

        system_msg = base_prompt.strip()
        if force_json_only:
            system_msg = f"{system_msg}\n\n# OUTPUT FORMAT (REQUIRED)\n{json_rules}"

        user_content = [
            {"type": "text", "text": "Aplica la matriz Marcus al caso siguiente y devuelve SOLO JSON válido:"},
            {"type": "text", "text": f"Contexto Marcus:\n{contexto_marcus}"},
            {"type": "text", "text": f"JSON Visual:\n{json_visual}"},
            {"type": "text", "text": f"JSON Transcripción:\n{json_transcripcion}"},
        ]

        mensajes = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_content),
        ]

        logging.info("📨 Enviando análisis de circunstancias Marcus al LLM (intento 1)...")
        respuesta = llm.invoke(mensajes)
        raw = respuesta.content if hasattr(respuesta, "content") else str(respuesta)

        parsed, err = _extract_json(raw)

        if parsed is not None and schema_validator:
            try:
                parsed = schema_validator(parsed)
            except Exception as sv_err:
                err = sv_err
                parsed = None

        attempts = 0
        while parsed is None and attempts < max_retries:
            attempts += 1
            logging.warning(f"🔁 Reintentando porque la respuesta no es JSON válido: {err}")

            fix_messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=[
                    {"type": "text",
                     "text": "La respuesta anterior NO fue JSON válido. Corrige y devuelve SOLO un JSON válido."},
                    {"type": "text",
                     "text": "RECUERDA: no incluyas texto adicional ni formato Markdown; solo el objeto JSON."},
                    {"type": "text", "text": f"Respuesta previa (para corregir):\n{raw}"},
                ])
            ]
            respuesta = llm.invoke(fix_messages)
            raw = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
            parsed, err = _extract_json(raw)

            if parsed is not None and schema_validator:
                try:
                    parsed = schema_validator(parsed)
                except Exception as sv_err:
                    err = sv_err
                    parsed = None

        if parsed is None:
            logging.error(f"❌ No se pudo obtener JSON válido del LLM: {err}")
            return {"error": f"No se pudo parsear JSON: {str(err)}", "raw": raw}

        # ✅ Devuelve el dict JSON directamente
        return parsed

    except Exception as e:
        logging.error(f"❌ Error al evaluar circunstancias Marcus: {e}", exc_info=True)
        return {"error": str(e)}
