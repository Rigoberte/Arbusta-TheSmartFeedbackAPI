from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model.logistic_regression_model import LogisticRegressionModel
from src.model.random_forest_model import RandomForestModel
from src.analyzer.sentiment_analyzer import SentimentAnalyzer
from src.settings import LOGISTIC_REGRESSION_MODEL_PATH, RANDOM_FOREST_MODEL_PATH, DATA_PATH

app = FastAPI(
    title="The Smart Feedback API",
    description="API para análisis de sentimiento de feedback",
    version="1.0.0"
)

class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"

# --- Request/Response Models ---
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Excelente servicio!"])
    model: ModelType = Field(default=ModelType.LOGISTIC_REGRESSION, description="Modelo a usar")

class AnalyzeResponse(BaseModel):
    sentiment: str
    score: float
    confidence: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    models_available: list[str]

# --- Model Loading ---
def get_analyzer(model_type: ModelType) -> SentimentAnalyzer:
    """Carga y retorna el analyzer con el modelo especificado."""
    if model_type == ModelType.LOGISTIC_REGRESSION:
        if not LOGISTIC_REGRESSION_MODEL_PATH.exists():
            LogisticRegressionModel.train(DATA_PATH, str(LOGISTIC_REGRESSION_MODEL_PATH))
        model = LogisticRegressionModel.load(LOGISTIC_REGRESSION_MODEL_PATH)
    else:
        if not RANDOM_FOREST_MODEL_PATH.exists():
            RandomForestModel.train(DATA_PATH, str(RANDOM_FOREST_MODEL_PATH))
        model = RandomForestModel.load(RANDOM_FOREST_MODEL_PATH)
    
    return SentimentAnalyzer(model)

# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Verifica el estado de la API."""
    return {
        "status": "ok",
        "models_available": [m.value for m in ModelType]
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_feedback(request: AnalyzeRequest):
    """
    Analiza el sentimiento de un texto.
    
    - text: Texto a analizar
    - model: Modelo a usar (logistic_regression o random_forest)
    
    Retorna:
    - sentiment: positivo, neutral o negativo
    - score: confianza del sentimiento predicho (0-1)
    - confidence: probabilidades de cada clase
    """
    try:
        analyzer = get_analyzer(request.model)
        result = analyzer.analyze(request.text)
        return {
            **result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_sentiment(request: AnalyzeRequest) -> dict:
    """
    Retorna solo el sentimiento (positivo, neutral, negativo).
    - text: Texto a analizar
    - model: Modelo a usar (logistic_regression o random_forest)

    Retorna:
    - sentiment: sentimiento predicho
    """
    try:
        analyzer = get_analyzer(request.model)
        return {
            "sentiment": analyzer.predict(request.text)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
