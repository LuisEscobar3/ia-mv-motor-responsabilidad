import os
import json
import dotenv
from langchain.globals import set_debug
from app.commons.services.llm_manager import load_llms
from app.commons.services.matrix_loader import cargar_matriz_marcus
from app.Funciones.procesar_audio import transcribir_audio_gemini
from app.Funciones.procesar_imagen import procesar_imagen
from app.Funciones.Procesar_circunstancias import evaluar_circunstancias_marcus
from app.Funciones.procesar_imagen import procesar_imagen_ficha

dotenv.load_dotenv()
set_debug(False)
os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")
llms = load_llms()
gemini = llms["gemini_pro"]

datos  = procesar_imagen_ficha(r"C:\Users\1032497498\PycharmProjects\Motor__responsabilidad\outputs\jose.luis.gomez@segurosbolivar.com - 10700053212\10700053212 captura.PNG",gemini)
print(datos)
