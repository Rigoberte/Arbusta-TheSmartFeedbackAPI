# The Smart Feedback API

Challenge técnico de Arbusta, donde se debe desarrollar un modelo que permita analizar comentarios de usuarios y agilizar la identificación y priorización de los casos críticos. Se debe diseñar, construir y entregar una aplicación en Python que exponga un servicio de análisis de sentimiento a través de una API RESTful.

## Requisitos
- Python 3.11+  
- Entorno virtual recomendado (`.venv`)

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno en Windows
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## Modos de Ejecución

El proyecto ofrece **3 formas diferentes** de interactuar con el analizador de sentimientos:

### 1. Interfaz Gráfica (GUI)

Interfaz visual moderna estilo ChatGPT para analizar sentimientos de forma interactiva.

```bash
python -m src.main_gui
```

---

### 2. API REST (FastAPI)
API RESTful para integrar el análisis de sentimientos en otras aplicaciones.

```bash
uvicorn src.main_api:app --reload
```

**Endpoints disponibles:**

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Verifica el estado de la API |
| POST | `/analyze` | Analiza el sentimiento de un texto |

**Documentación interactiva:**  http://localhost:8000/docs

---

### 3. 💻 Línea de Comandos (CLI)

Interfaz de terminal para análisis rápido de textos.

```bash
python -m src.main_cli
```

**Comandos disponibles:**

| Comando | Descripción |
|---------|-------------|
| `exit` | Salir del programa |
| `change-model-to-rf` | Cambiar a modelo Random Forest |
| `change-model-to-lr` | Cambiar a modelo Regresión Logística |

---

## Modelos Disponibles

| Modelo | Descripción |
|--------|-------------|
| **Regresión Logística** | Modelo por defecto, rápido y eficiente |
| **Random Forest** | Mayor precisión en algunos casos |

Los modelos se entrenan automáticamente la primera vez que se utilizan con los datos de `data/reviews.csv`.

---

## Tests

```bash
pytest tests/ -v
```

---

## Estructura del Proyecto

```
├── src/
│   ├── main_api.py # API REST con FastAPI
│   ├── main_cli.py # Interfaz con consola
│   ├── main_gui.py # Interfaz con CustomTkinter
│   ├── analyzer/
│   │   └── sentiment_analyzer.py
│   └── model/
│       ├── base.py
│       ├── logistic_regression_model.py
│       └── random_forest_model.py
├── data/
│   └── reviews.csv
├── tests/
│   ├── test_sentiment_analyzer.py
│   └── test_sklearn_model.py
├── requirements.txt
└── README.md
```
