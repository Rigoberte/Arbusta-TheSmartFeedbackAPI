from typing import Dict

from src.model.base import Model

class SentimentAnalyzer:
    SENTIMENTS = ['positivo', 'neutral', 'negativo']
    
    def __init__(self, model: Model):
        """
        Args:
            model: Cualquier implementación de Model (sklearn, transformers, etc.)
        """
        self._model = model
    
    def analyze(self, text: str) -> Dict:
        self._validate_text(text)
        
        sentiment = self._model.predict([text])[0]
        probas = self._get_probabilities(text)
        
        return {
            'sentiment': sentiment,
            'score': probas[sentiment],
            'confidence': probas
        }
    
    def predict(self, text: str) -> str:
        self._validate_text(text)
        return self._model.predict([text])[0]
    
    def _get_probabilities(self, text: str) -> Dict[str, float]:
        probas = self._model.predict_proba([text])[0]
        return {
            class_name: float(proba) 
            for class_name, proba in zip(self.SENTIMENTS, probas)
        }
    
    def _validate_text(self, text: str) -> None:
        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")