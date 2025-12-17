import pandas as pd

def cargar_matriz_marcus(ruta_excel: str, hoja: str = "Descripción") -> str:
    """
    Carga la hoja de circunstancias del Excel de Marcus y genera un string legible como contexto para el LLM.
    Incluye también el Código Nacional de Tránsito como parte del razonamiento normativo.
    """

    # Leer la hoja
    df = pd.read_excel(ruta_excel, sheet_name=hoja)

    # Renombrar columnas para estandarizar
    df = df.rename(columns={
        "CIRCUNSTANCIAS": "id",
        "CODIGO NACIONAL DE TRANSITO": "codigo",
        "DESCRIPCION CESVI": "descripcion"
    })

    # Validación
    if not {"id", "codigo", "descripcion"}.issubset(df.columns):
        raise ValueError("La hoja debe contener las columnas 'CIRCUNSTANCIAS', 'CÓDIGO NACIONAL DE TRÁNSITO' y 'DESCRIPCION CESVI'.")

    # Crear texto de contexto
    texto_contexto = "Estas son las 15 circunstancias definidas por Marcus, con su justificación legal y técnica:\n\n"
    for _, fila in df.iterrows():
        texto_contexto += (
            f"{fila['id']}:\n"
            f"- Descripción CESVI: {fila['descripcion']}\n"
            f"- Fundamento legal (Código Nacional de Tránsito): {fila['codigo']}\n\n"
        )

    return texto_contexto.strip()
