## transcripciones-danos-bienes-terceros-movilidad-ms

### Generalidades

Servicio para hacer analisis tecnico de llamadas (audio) y transcripciones (texto) de negociaciones de daños de bienes a terceros (DBT).

#### Carpetas

```
audios
transcripciones
app
├── __init__.py
├── commons
│   ├── __init__.py
│   ├── dto
│   │   └── __init__.py
│   ├── exceptions
│   │   └── __init__.py
│   └── services
│       ├── __init__.py
│       └── miscelaneous.py
├── config
│   ├── __init__.py
│   └── llm_parameters.json
├── procesar_transcripcion
│   ├── __init__.py
│   ├── controller
│   ├── models
│   └── services
├── routers
│   └── __init__.py
└── utils
    ├── __init__.py
    └── prompts_generales.yaml
test_procesar_audios.py
test_procesar_transcripcion.py
```

+ **app:** Carpeta raiz del paquete del servicio
+ **commons:** Servicios y datos transversales para todo el paquete
+ **config:** Contiene los archivos de configuración, `llm_parameters.json` parametros para acceso a modelos LLM.
+ **procesar_trasncripcion:** Contiene la logica del servicio ligada al procesamiento de audios/trasncripciones.
+ **utils:** Archivos utilitarios. `prompts_generales.yaml` contiene la estructura de los prompts usandos.
+ **audios:** Carpeta con ejemplos de audios de conversaciones de DBT.
+ **transcripciones:** Carpeta con ejemplos de transcripciones de accidentes autos.
+ **root:** `test_procesar_audios.py` propotitpo basico de extracción de datos a partir de un archivo de audio a través de Gemini. `test_procesar_transcripcion.py` prototipo basico de extracción de datos a partir del texto de una transcripcion usando GPT4-o-mini.


### Requisistos

El archivo `requirements.txt` contiene los paquetes necesarios para la ejecución del servicio.

El archivo `langchain.txt` contiene los paquetes especificos relacionados con langchain.

#### Variables de entorno:

```
APP_ENV - Entorno de ejecución de libreria de langchain
```