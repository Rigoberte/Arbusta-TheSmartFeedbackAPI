from pathlib import Path

from src.model.sklearn_model import SklearnModel
from src.analyzer.sentiment_analyzer import SentimentAnalyzer


MODEL_PATH = Path("src/model/sentiment_model.pkl")
DATA_PATH = "data/reviews.csv"


def main():
    if not MODEL_PATH.exists():
        print("Creando modelo...")
        SklearnModel.train(DATA_PATH, str(MODEL_PATH))
        print("Modelo creado!\n")
    
    model = SklearnModel.load(MODEL_PATH)
    analyzer = SentimentAnalyzer(model)
    
    print("Analizador de sentimiento (escribe 'exit' para salir)\n")
    
    while True:
        text = input("Comentario: ").strip()
        
        if text.lower() == "exit":
            break
        
        if not text:
            continue
        
        result = analyzer.analyze(text)
        print(f"  > {result['sentiment']} ({result['score']:.1%})\n")

if __name__ == "__main__":
    main()
