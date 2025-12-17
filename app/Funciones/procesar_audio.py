import mimetypes
import logging


from app.commons.services.miscelaneous import load_prompts_generales
from langchain_core.messages import SystemMessage, HumanMessage


def transcribir_audio_gemini(ruta_audio: str, llm) -> str:
    """
    Transcribe un archivo de audio usando Gemini multimodal con prompt cargado desde YAML.

    Args:
        ruta_audio: Ruta local del archivo de audio.
        llm: Objeto LLM ya inicializado (ej. Gemini Pro)

    Returns:
        Texto transcrito como string limpio.
    """
    try:
        system_prompt = load_prompts_generales("transcription_audio")
        if not system_prompt:
            raise ValueError("❌ Prompt 'transcripcion_audio' no encontrado en YAML.")

            # 2. Determinar el tipo MIME del audio
        mime_type, _ = mimetypes.guess_type(ruta_audio)
        if not mime_type:
            raise ValueError("No se pudo determinar el tipo MIME del archivo de audio.")

        with open(ruta_audio, "rb") as f:
            audio_content = f.read()

        # 3. Estructurar mensaje multimodal
        messages_for_llm: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": "Procesa el siguiente audio según las instrucciones del sistema:"},
                {"type": "media", "data": audio_content, "mime_type": mime_type}
            ])
        ]

        logging.info(f"🎙️ Enviando audio '{ruta_audio}' a Gemini para transcripción...")

        # 4. Invocar el modelo
        response_obj = llm.invoke(messages_for_llm)
        xml_response = response_obj.content if hasattr(response_obj, "content") else str(response_obj)

        # 5. Extraer texto entre <TRANSCRIPCION>...</TRANSCRIPCION>
        return xml_response

    except Exception as e:
        logging.error(f"❌ Error durante la transcripción con Gemini: {e}")
        return ""
