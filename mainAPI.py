import os
import json
import uuid
import dotenv
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.concurrency import run_in_threadpool

# LangChain import compatible (old/new)
try:
    from langchain.globals import set_debug
except Exception:
    from langchain_core.globals import set_debug

# --- TU PROYECTO ---
from app.commons.services.llm_manager import load_llms
from app.commons.services.matrix_loader import cargar_matriz_marcus

from app.Funciones.procesar_audio import transcribir_audio_gemini
from app.Funciones.procesar_imagen import procesar_imagen, procesar_imagen_ficha
from app.Funciones.Procesar_circunstancias import evaluar_circunstancias_marcus
from app.Funciones.presicion import evaluar_coherencia_visual_vs_ficha


# ============================================================
# STORAGE (Cloud Run friendly)
# ============================================================
BASE_DIR = Path(os.environ.get("WORKDIR", "/tmp/motor_resp"))
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXT_VISUAL = {".pdf"}
EXT_FICHA = {".png"}
EXT_AUDIO = {".mp3", ".wav", ".m4a", ".ogg"}


def _save_json(data: Any, path_file: Path):
    path_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_text(text: str, path_file: Path):
    path_file.write_text(text if isinstance(text, str) else str(text), encoding="utf-8")


def _validate_ext(filename: str, allowed: set, label: str):
    if not filename:
        raise HTTPException(400, f"Falta archivo: {label}")
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Archivo inválido para {label}. Ext permitidas: {sorted(list(allowed))}")


# ============================================================
# LIFESPAN: carga LLM + matriz 1 sola vez
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    dotenv.load_dotenv()
    set_debug(False)
    os.environ["APP_ENV"] = os.environ.get("APP_ENV", "sbx")

    # 1) LLM
    llms = load_llms()
    gemini = llms.get("gemini_pro")
    if not gemini:
        raise RuntimeError("No se pudo cargar gemini_pro desde load_llms()")
    app.state.gemini = gemini

    # 2) Matriz Marcus
    # En Cloud Run, usa ruta relativa dentro del repo/imagen o una env var:
    # export MARCUS_XLSX_PATH=/app/app/utils/Descripción Circunstancias.xlsx
    marcus_path = os.environ.get("MARCUS_XLSX_PATH", "app/utils/Descripción Circunstancias.xlsx")
    if not Path(marcus_path).exists():
        raise RuntimeError(f"No se encontró el Excel Marcus en: {marcus_path}")
    app.state.contexto_marcus = cargar_matriz_marcus(marcus_path)

    yield


app = FastAPI(title="Motor Responsabilidad API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}


# ============================================================
# CORE: tu pipeline (mismas 5 fases)
# ============================================================
def _procesar_caso_por_rutas(
    case_id: str,
    ruta_visual_pdf: str,
    ruta_ficha_png: str,
    ruta_audio: str,
    gemini,
    contexto_marcus,
) -> Dict[str, Any]:
    out_case_dir = OUTPUT_DIR / case_id
    out_case_dir.mkdir(parents=True, exist_ok=True)

    # 1) ANALISIS VISUAL
    hechos_visual = procesar_imagen(ruta_visual_pdf, llm=gemini)
    _save_json(hechos_visual, out_case_dir / "hechos_visual.json")

    # 2) FICHA DEL SINIESTRO (PNG)
    ficha_siniestro = procesar_imagen_ficha(ruta_ficha_png, gemini)
    _save_json(ficha_siniestro, out_case_dir / "ficha_siniestro.json")

    # 3) TRANSCRIPCIÓN AUDIO
    texto_transcrito = transcribir_audio_gemini(ruta_audio, llm=gemini)
    _save_text(texto_transcrito, out_case_dir / "transcripcion.txt")

    # 4) EVALUAR CIRCUNSTANCIAS MARCUS
    resultado_circunstancias = evaluar_circunstancias_marcus(
        contexto_marcus=contexto_marcus,
        json_visual=hechos_visual.get("resultado", hechos_visual),
        json_transcripcion=texto_transcrito,
        llm=gemini,
    )
    _save_json(resultado_circunstancias, out_case_dir / "resultado_circunstancias.json")

    # 5) EVALUAR PRECISIÓN VISUAL VS FICHA
    resultado_precision = evaluar_coherencia_visual_vs_ficha(
        llm=gemini,
        json_analisis_visual=json.dumps(hechos_visual, ensure_ascii=False),
        json_ficha_siniestro=json.dumps(ficha_siniestro, ensure_ascii=False),
    )
    _save_json(resultado_precision, out_case_dir / "precision_visual_vs_ficha.json")

    return {
        "case_id": case_id,
        "hechos_visual": hechos_visual,
        "ficha_siniestro": ficha_siniestro,
        "transcripcion_text": texto_transcrito,
        "resultado_circunstancias": resultado_circunstancias,
        "precision_visual_vs_ficha": resultado_precision,
        "outputs_dir": str(out_case_dir),
    }


# ============================================================
# ENDPOINT: recibe 3 archivos y procesa 1 caso
# ============================================================
@app.post("/process-case")
async def process_case(
    visual_pdf: UploadFile = File(...),
    ficha_png: UploadFile = File(...),
    audio: UploadFile = File(...),
    case_id: Optional[str] = Form(None),  # opcional, si no lo mandas se genera
):
    _validate_ext(visual_pdf.filename, EXT_VISUAL, "visual_pdf")
    _validate_ext(ficha_png.filename, EXT_FICHA, "ficha_png")
    _validate_ext(audio.filename, EXT_AUDIO, "audio")

    case_id = case_id or uuid.uuid4().hex

    # Guardar en /tmp (Cloud Run)
    pdf_path = UPLOAD_DIR / f"{case_id}_visual{Path(visual_pdf.filename).suffix.lower()}"
    png_path = UPLOAD_DIR / f"{case_id}_ficha{Path(ficha_png.filename).suffix.lower()}"
    aud_path = UPLOAD_DIR / f"{case_id}_audio{Path(audio.filename).suffix.lower()}"

    pdf_path.write_bytes(await visual_pdf.read())
    png_path.write_bytes(await ficha_png.read())
    aud_path.write_bytes(await audio.read())

    gemini = app.state.gemini
    contexto_marcus = app.state.contexto_marcus

    try:
        result = await run_in_threadpool(
            _procesar_caso_por_rutas,
            case_id,
            str(pdf_path),
            str(png_path),
            str(aud_path),
            gemini,
            contexto_marcus,
        )
    except Exception as e:
        raise HTTPException(500, f"Error procesando caso {case_id}: {e}")

    return {"ok": True, **result}
