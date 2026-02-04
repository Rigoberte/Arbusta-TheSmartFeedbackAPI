from pathlib import Path

from src.model.logistic_regression_model import LogisticRegressionModel
from src.model.random_forest_model import RandomForestModel
from src.analyzer.sentiment_analyzer import SentimentAnalyzer


LOGISTIC_REGRESSION_MODEL_PATH = Path("src/model/logistic_regression_model.pkl")
RANDOM_FOREST_MODEL_PATH = Path("src/model/random_forest_model.pkl")
DATA_PATH = "data/reviews.csv"


def main():
    print("The Smart Feedback")
    print("> Escribe 'exit' para salir\n")
    print("> Escribe 'change-model-to-rf' para cambiar al modelo con Random Forest\n")
    print("> Escribe 'change-model-to-lr' para cambiar al modelo con Regresión Logística\n")
    
    analyzer = create_sentiment_analyzer_with_logistic_regression() # Modelo por defecto
    
    while True:
        text = input("Comentario: ").strip()
        
        if text.lower() == "exit":
            break

        if text.lower() == "change-model-to-rf":
            analyzer = create_sentiment_analyzer_with_random_forest()
            print("Modelo cambiado a Random Forest.\n")
            continue
        
        if text.lower() == "change-model-to-lr":
            analyzer = create_sentiment_analyzer_with_logistic_regression()
            print("Modelo cambiado a Regresión Logística.\n")
            continue
        
        if not text:
            continue
        
        result = analyzer.analyze(text)
        print(f"  > {result['sentiment']} ({result['score']:.1%})\n")

def create_sentiment_analyzer_with_logistic_regression() -> SentimentAnalyzer:
    if not LOGISTIC_REGRESSION_MODEL_PATH.exists():
        LogisticRegressionModel.train(DATA_PATH, str(LOGISTIC_REGRESSION_MODEL_PATH))
        
    model = LogisticRegressionModel.load(LOGISTIC_REGRESSION_MODEL_PATH)
    return SentimentAnalyzer(model)

def create_sentiment_analyzer_with_random_forest() -> SentimentAnalyzer:
    if not RANDOM_FOREST_MODEL_PATH.exists():
        RandomForestModel.train(DATA_PATH, str(RANDOM_FOREST_MODEL_PATH))
        
    model = RandomForestModel.load(RANDOM_FOREST_MODEL_PATH)
    return SentimentAnalyzer(model)

if __name__ == "__main__":
    main()