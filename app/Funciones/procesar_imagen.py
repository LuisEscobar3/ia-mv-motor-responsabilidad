import os
import json
import logging
import mimetypes
from typing import List, Dict, Tuple, Any

import fitz  # PyMuPDF
from langchain_core.messages import SystemMessage, HumanMessage

from app.commons.services.miscelaneous import load_prompts_generales

# =========================
# Config básica de logging
# =========================
# Ajusta el nivel si quieres menos/más verbosidad
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(asctime)s - %(message)s"
)


# =========================
# Utilidades JSON / Normalización
# =========================
def _clean_markdown_fences(texto: str) -> str:
    """
    Elimina fences de Markdown (```json ... ``` o ``` ... ```) si el modelo los incluyó.
    """
    if not isinstance(texto, str):
        return str(texto)
    t = texto.strip()
    if t.startswith("```json"):
        t = t.removeprefix("```json").strip()
        if t.endswith("```"):
            t = t.removesuffix("```").strip()
        return t
    if t.startswith("```"):
        t = t.removeprefix("```").strip()
        if t.endswith("```"):
            t = t.removesuffix("```").strip()
        return t
    return t


def _normalize_schema(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Acepta variantes de clave del modelo y las mapea al contrato esperado.
    Para tu prompt actual, el contrato usa 'inferencias_tecnicas'.
    """
    # Si viniera de versiones previas con 'inferencias_preliminares', mapea -> inferencias_tecnicas
    if "inferencias_preliminares" in d and "inferencias_tecnicas" not in d:
        d["inferencias_tecnicas"] = d.pop("inferencias_preliminares")

    # También aceptamos 'observaciones' -> 'observaciones_objetivas' si alguna iteración del prompt lo cambió
    if "observaciones" in d and "observaciones_objetivas" not in d:
        d["observaciones_objetivas"] = d.pop("observaciones")

    return d


def _validate_schema(d: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Valida presencia de claves esenciales del contrato.
    (No confundir con JSON malformado: esto es esquema inesperado.)
    """
    required = [
        "metadata_analisis",
        "observaciones_objetivas",
        "inferencias_tecnicas",
        "limitaciones_y_incertidumbres",
    ]
    missing = [k for k in required if k not in d]
    if missing:
        return False, f"Faltan claves esperadas: {', '.join(missing)}"
    return True, ""


# =========================
# Conversión de PDF a JPG
# =========================
def convertir_pdf_a_jpgs(pdf_path: str, dpi: int = 150) -> List[str]:
    """
    Convierte todas las páginas de un PDF a imágenes JPG usando PyMuPDF.
    Devuelve una lista de rutas de salida.
    """
    rutas: List[str] = []
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"No existe el archivo PDF: {pdf_path}")

    with fitz.open(pdf_path) as doc:  # asegura cierre del documento
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)  # 150 dpi = buen balance calidad/memoria
            salida = pdf_path.replace(".pdf", f"_page{i + 1}.jpg")
            pix.save(salida)
            rutas.append(salida)

    logging.info(f"📄 PDF convertido a {len(rutas)} JPG(s).")
    return rutas


# =========================
# Detección MIME robusta
# =========================
def _guess_mime(ruta: str) -> str:
    mime_type, _ = mimetypes.guess_type(ruta)
    if mime_type:
        return mime_type
    ext = os.path.splitext(ruta)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
        ".pdf": "application/pdf",
    }.get(ext, "application/octet-stream")


# =========================
# Procesamiento principal
# =========================
def procesar_imagen(ruta_archivo: str, llm) -> Dict[str, Any]:
    """
    Analiza una imagen o PDF con Gemini multimodal en un solo prompt.
    Devuelve un dict con la estructura estándar o error.

    Args:
        ruta_archivo: Ruta al archivo .jpg, .png o .pdf
        llm: Objeto LangChain LLM multimodal (ya configurado)

    Returns:
        dict: { "archivo": str, "resultado": dict } en éxito,
              o { "error": str, ... } en fallo.
    """
    try:
        if not os.path.isfile(ruta_archivo):
            return {"error": f"Ruta no válida o archivo no existe: {ruta_archivo}"}

        prompt = load_prompts_generales("extraction_visual")
        if not prompt:
            raise ValueError("❌ Prompt 'extraction_visual' no encontrado en YAML.")

        # Convertir si es PDF
        if ruta_archivo.lower().endswith(".pdf"):
            logging.info(f"📄 Convirtiendo PDF '{ruta_archivo}' a imágenes JPG con PyMuPDF...")
            rutas_imagenes = convertir_pdf_a_jpgs(ruta_archivo, dpi=150)
            if not rutas_imagenes:
                return {"error": "No se generaron imágenes a partir del PDF."}
        else:
            rutas_imagenes = [ruta_archivo]

        # Construir bloques multimedia
        media_blocks: List[Dict[str, Any]] = []
        for ruta_img in rutas_imagenes:
            mime_type = _guess_mime(ruta_img)
            try:
                with open(ruta_img, "rb") as f:
                    data = f.read()
            except Exception as e:
                logging.error(f"❌ No se pudo leer '{ruta_img}': {e}")
                return {"error": f"No se pudo leer la imagen: {ruta_img}", "detalle": str(e)}

            media_blocks.append({
                "type": "media",
                "data": data,
                "mime_type": mime_type
            })

        # Construir mensajes para LLM
        messages_for_llm = [
            SystemMessage(content=prompt),
            HumanMessage(content=[{"type": "text",
                                   "text": "Analiza todas las imágenes siguientes según el formato establecido:"}] + media_blocks)
        ]

        logging.info("🧠 Enviando imágenes al LLM para análisis visual consolidado...")
        respuesta = llm.invoke(messages_for_llm)

        # Extraer texto devolviendo siempre str
        texto = getattr(respuesta, "content", None)
        if texto is None:
            texto = str(respuesta)

        # Limpieza de fences
        texto = _clean_markdown_fences(texto)

        # Parseo JSON
        try:
            json_resultado = json.loads(texto)
        except json.JSONDecodeError as e:
            lines = texto.splitlines()
            line_number = getattr(e, "lineno", -1)
            col_number = getattr(e, "colno", -1)
            error_line = lines[line_number - 1] if 0 < line_number <= len(lines) else "<línea no encontrada>"

            logging.warning(
                f"⚠️ JSON MALFORMADO en línea {line_number}, columna {col_number}: {e}\n"
                f"🧾 Línea problemática:\n{error_line}\n"
                f"🔍 Texto completo:\n{texto}"
            )
            return {
                "error": "Respuesta no es JSON válido (mal formado)",
                "lineno": line_number,
                "colno": col_number,
                "raw_response": texto
            }

        # Normalización de esquema y validación
        json_resultado = _normalize_schema(json_resultado)
        ok, msg = _validate_schema(json_resultado)
        if not ok:
            logging.warning(
                f"⚠️ Esquema inesperado: {msg}\n🔍 JSON recibido (truncado): {json.dumps(json_resultado, ensure_ascii=False)[:2000]}...")
            return {
                "error": "Respuesta JSON válida pero con esquema inesperado",
                "schema_issue": msg,
                "raw_response": json_resultado
            }

        # Éxito
        logging.info("✅ Análisis visual estructurado recibido correctamente.")
        return {
            "archivo": os.path.basename(ruta_archivo),
            "resultado": json_resultado
        }

    except Exception as e:
        logging.error(f"❌ Error en procesar_imagen: {e}")
        return {"error": str(e)}


def procesar_imagen_ficha(ruta_archivo: str, llm) -> Dict[str, Any]:
    """
    Envía una imagen o PDF al LLM con el prompt 'extraction_visual_Ficha'
    y devuelve directamente el JSON estructurado que responde el modelo.
    """
    import os, json, logging
    from typing import Dict, Any, List
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        # Validar archivo
        if not os.path.isfile(ruta_archivo):
            return {"error": f"Ruta no válida o archivo no existe: {ruta_archivo}"}

        # Cargar prompt
        prompt = load_prompts_generales("extraction_visual_Ficha")
        if not prompt:
            raise ValueError("❌ Prompt 'extraction_visual_Ficha' no encontrado en YAML.")

        # Si es PDF → convertir a JPG
        if ruta_archivo.lower().endswith(".pdf"):
            rutas_imagenes = convertir_pdf_a_jpgs(ruta_archivo, dpi=150)
            if not rutas_imagenes:
                return {"error": "No se generaron imágenes a partir del PDF."}
        else:
            rutas_imagenes = [ruta_archivo]

        # Construir bloques multimedia
        media_blocks: List[Dict[str, Any]] = []
        for ruta_img in rutas_imagenes:
            mime_type = _guess_mime(ruta_img)
            with open(ruta_img, "rb") as f:
                data = f.read()
            media_blocks.append({"type": "media", "data": data, "mime_type": mime_type})

        # Preparar mensajes
        messages_for_llm = [
            SystemMessage(content=prompt),
            HumanMessage(content=[{"type": "text", "text": "Extrae exactamente el JSON solicitado:"}] + media_blocks)
        ]

        # Llamar al modelo
        respuesta = llm.invoke(messages_for_llm)
        texto = getattr(respuesta, "content", None)
        if texto is None:
            texto = str(respuesta)

        # Limpiar fences markdown
        texto = _clean_markdown_fences(texto)

        # Parsear y devolver directamente el JSON
        return json.loads(texto)

    except json.JSONDecodeError as e:
        return {
            "error": "Respuesta no es JSON válido (mal formado)",
            "detalle": str(e),
        }
    except Exception as e:
        logging.error(f"❌ Error en procesar_imagen_ficha: {e}")
        return {"error": str(e)}
