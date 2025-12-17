import json
import logging
from typing import Callable, Optional, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from app.commons.services.miscelaneous import load_prompts_generales


def _strip_code_fences(text: str) -> str:
    """
    Elimina fences de código tipo ``` de una respuesta del LLM.
    """
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
    """
    Intenta extraer JSON válido desde un string.
    - Primero intenta json.loads directo.
    - Luego sin fences.
    - Luego buscando el primer '{' y el último '}'.
    Devuelve (objeto_json, error) donde error es None si todo salió bien.
    """
    if text is None:
        return None, ValueError("Empty response")

    # Intento directo
    try:
        return json.loads(text), None
    except Exception:
        pass

    # Intento sin fences
    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped), None
    except Exception:
        pass

    # Intento recortando desde el primer '{' hasta el último '}'
    try:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start:end + 1]
            return json.loads(candidate), None
    except Exception as e:
        return None, e

    return None, ValueError("Could not parse JSON after repairs")


def evaluar_coherencia_visual_vs_ficha(
        llm: object,
        json_analisis_visual: str,
        json_ficha_siniestro: str,
        *,
        schema_validator: Optional[Callable[[Any], Any]] = None,
        schema_description: Optional[str] = None,
        force_json_only: bool = True,
        max_retries: int = 1,
) -> Any:
    """
    Evalúa la coherencia entre:
    - ANÁLISIS VISUAL (json_analisis_visual)
    - FICHA DEL SINIESTRO (json_ficha_siniestro)

    Envía ambos JSON al LLM usando el prompt:
    - 'evaluar_coherencia_visual_vs_ficha' (cargado desde YAML)

    Garantiza que la salida sea SIEMPRE un dict (JSON parseado).
    En caso de error devuelve {"error": "..."}.
    """
    try:
        # 1. Cargar prompt base desde YAML
        base_prompt = load_prompts_generales("evcaluacion_presicion_")
        if not base_prompt:
            return {"error": "❌ Prompt 'evaluar_coherencia_visual_vs_ficha' no encontrado en YAML."}

        # 2. Reglas estrictas de salida JSON
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

        # 3. Construir mensaje de usuario (SOLO DOS JSON, como pediste)
        user_content = [
            {
                "type": "text",
                "text": (
                    "Evalúa la coherencia entre el ANÁLISIS VISUAL del siniestro y la FICHA DOCUMENTAL, "
                    "asociando placas, interpretando la causa del siniestro y comparando la responsabilidad. "
                    "Devuelve SOLO un JSON válido siguiendo las instrucciones del sistema."
                ),
            },
            {"type": "text", "text": "ANÁLISIS VISUAL (JSON):"},
            {"type": "text", "text": json_analisis_visual},
            {"type": "text", "text": "FICHA DEL SINIESTRO (JSON):"},
            {"type": "text", "text": json_ficha_siniestro},
        ]

        mensajes = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_content),
        ]

        # 4. Primera invocación al LLM
        logging.info("📨 Enviando evaluación de coherencia visual vs ficha al LLM (intento 1)...")
        respuesta = llm.invoke(mensajes)
        raw = respuesta.content if hasattr(respuesta, "content") else str(respuesta)

        parsed, err = _extract_json(raw)

        # 5. Validación opcional contra schema (Pydantic u otro)
        if parsed is not None and schema_validator:
            try:
                parsed = schema_validator(parsed)
            except Exception as sv_err:
                err = sv_err
                parsed = None

        # 6. Reintentos si no se obtuvo JSON válido
        attempts = 0
        while parsed is None and attempts < max_retries:
            attempts += 1
            logging.warning(f"🔁 Reintentando porque la respuesta no es JSON válido: {err}")

            fix_messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": (
                            "La respuesta anterior NO fue JSON válido. Corrige y devuelve SOLO un JSON válido. "
                            "RECUERDA: no incluyas texto adicional ni formato Markdown; solo el objeto JSON."
                        ),
                    },
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

        # 7. Manejo de fallo definitivo
        if parsed is None:
            logging.error(f"❌ No se pudo obtener JSON válido del LLM: {err}")
            return {"error": f"No se pudo parsear JSON: {str(err)}", "raw": raw}

        # ✅ Devuelve el dict JSON directamente
        return parsed

    except Exception as e:
        logging.error(f"❌ Error al evaluar coherencia visual vs ficha: {e}", exc_info=True)
        return {"error": str(e)}
