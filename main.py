import os
import json
import dotenv
from langchain.globals import set_debug

from app.commons.services.llm_manager import load_llms
from app.commons.services.matrix_loader import cargar_matriz_marcus

from app.Funciones.procesar_audio import transcribir_audio_gemini
from app.Funciones.procesar_imagen import procesar_imagen, procesar_imagen_ficha
from app.Funciones.Procesar_circunstancias import evaluar_circunstancias_marcus
from app.Funciones.presicion import evaluar_coherencia_visual_vs_ficha


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

dotenv.load_dotenv()
set_debug(False)
os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")

llms = load_llms()
gemini = llms["gemini_pro"]

ruta_excel_marcus = r"app/utils/Descripción Circunstancias.xlsx"
contexto_marcus = cargar_matriz_marcus(ruta_excel_marcus)


# ============================================================
# EXTENSIONES PERMITIDAS
# ============================================================

EXT_VISUAL = {".pdf"}             # imagen para análisis visual
EXT_FICHA = {".png"}              # imagen para FICHA DEL SINIESTRO
EXT_AUDIO = {".mp3", ".wav", ".m4a", ".ogg"}


# ============================================================
# UTILIDADES
# ============================================================

def _listar_archivos_por_extension(directorio, extensiones):
    archivos = []
    try:
        for nombre in sorted(os.listdir(directorio)):
            ruta = os.path.join(directorio, nombre)
            if os.path.isfile(ruta):
                _, ext = os.path.splitext(nombre)
                if ext.lower() in extensiones:
                    archivos.append(ruta)
    except FileNotFoundError:
        pass
    return archivos


def _ensure_dir(path_dir):
    os.makedirs(path_dir, exist_ok=True)


def _save_json(data, path_file):
    try:
        with open(path_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Guardado JSON: {path_file}")
    except Exception as e:
        print(f"⚠️ No se pudo guardar {path_file}: {e}")


def _save_text(text, path_file):
    try:
        with open(path_file, "w", encoding="utf-8") as f:
            f.write(text if isinstance(text, str) else str(text))
        print(f"💾 Guardado TXT: {path_file}")
    except Exception as e:
        print(f"⚠️ No se pudo guardar {path_file}: {e}")


# ============================================================
# DIRECTORIOS PRINCIPALES
# ============================================================

raiz_casos = r"./inputs"
out_root = r"C:\Users\1032497498\PycharmProjects\Motor__responsabilidad\outputs"


# ============================================================
# LOOP PRINCIPAL
# ============================================================

if not os.path.isdir(raiz_casos):
    raise FileNotFoundError(f"Directorio raíz no encontrado: {raiz_casos}")

subdirectorios = [
    os.path.join(raiz_casos, d)
    for d in sorted(os.listdir(raiz_casos))
    if os.path.isdir(os.path.join(raiz_casos, d))
]

if not subdirectorios:
    print(f"No se encontraron casos en: {raiz_casos}")


for dir_caso in subdirectorios:
    nombre_caso = os.path.basename(dir_caso)
    print(f"\n================= CASO: {nombre_caso} =================")

    # Buscar archivos específicos
    visual_pdf = _listar_archivos_por_extension(dir_caso, EXT_VISUAL)
    ficha_png = _listar_archivos_por_extension(dir_caso, EXT_FICHA)
    audios = _listar_archivos_por_extension(dir_caso, EXT_AUDIO)

    if not visual_pdf:
        print(f"⚠️  Sin PDF para análisis visual en {nombre_caso}. Se omite.")
        continue

    if not ficha_png:
        print(f"⚠️  Sin PNG para la ficha del siniestro en {nombre_caso}. Se omite.")
        continue

    if not audios:
        print(f"⚠️  Sin audio en {nombre_caso}. Se omite.")
        continue

    ruta_visual = visual_pdf[0]
    ruta_ficha = ficha_png[0]
    ruta_audio = audios[0]

    # Carpeta salida
    out_case_dir = os.path.join(out_root, nombre_caso)
    _ensure_dir(out_case_dir)

    # ============================================================
    # 1) ANALISIS VISUAL
    # ============================================================
    hechos_visual = procesar_imagen(ruta_visual, llm=gemini)
    print("🖼️ Hechos visuales extraídos OK.")
    _save_json(hechos_visual, os.path.join(out_case_dir, "hechos_visual.json"))

    # ============================================================
    # 2) FICHA DEL SINIESTRO (PNG)
    # ============================================================
    ficha_siniestro = procesar_imagen_ficha(ruta_ficha, gemini)
    print("📄 Ficha del siniestro extraída OK.")
    _save_json(ficha_siniestro, os.path.join(out_case_dir, "ficha_siniestro.json"))

    # ============================================================
    # 3) TRANSCRIPCIÓN AUDIO
    # ============================================================
    texto_transcrito = transcribir_audio_gemini(ruta_audio, llm=gemini)
    print("🗣️ Transcripción OK.")
    _save_text(texto_transcrito, os.path.join(out_case_dir, "transcripcion.txt"))

    # ============================================================
    # 4) EVALUAR CIRCUNSTANCIAS MARCUS
    # ============================================================
    resultado_circunstancias = evaluar_circunstancias_marcus(
        contexto_marcus=contexto_marcus,
        json_visual=hechos_visual.get("resultado", hechos_visual),
        json_transcripcion=texto_transcrito,
        llm=gemini
    )
    print("📘 Resultado circunstancias OK.")
    _save_json(resultado_circunstancias, os.path.join(out_case_dir, "resultado_circunstancias.json"))

    # ============================================================
    # 5) EVALUAR PRECISIÓN VISUAL VS FICHA
    # ============================================================
    resultado_precision = evaluar_coherencia_visual_vs_ficha(
        llm=gemini,
        json_analisis_visual=json.dumps(hechos_visual),
        json_ficha_siniestro=json.dumps(ficha_siniestro)
    )

    print("🎯 Evaluación de precisión OK.")
    _save_json(resultado_precision, os.path.join(out_case_dir, "precision_visual_vs_ficha.json"))
