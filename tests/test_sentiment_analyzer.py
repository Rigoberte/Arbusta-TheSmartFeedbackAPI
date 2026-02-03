import pytest
from typing import List

from src.model.base import Model
from src.analyzer.sentiment_analyzer import SentimentAnalyzer


class FakeModel(Model):
    """Mock de Model para tests"""
    
    def __init__(self, sentiment: str = "positivo", probas: List[float] = None):
        self._sentiment = sentiment
        self._probas = probas or [0.8, 0.15, 0.05]
    
    def predict(self, texts: List[str]) -> List[str]:
        return [self._sentiment] * len(texts)
    
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        return [self._probas] * len(texts)


class TestSentimentAnalyzer:
    def test_predict_returns_sentiment(self):
        model = FakeModel(sentiment="positivo")
        analyzer = SentimentAnalyzer(model)
        
        result = analyzer.predict("Me encanta este producto")
        
        assert result == "positivo"
    
    def test_predict_negative_sentiment(self):
        model = FakeModel(sentiment="negativo")
        analyzer = SentimentAnalyzer(model)
        
        result = analyzer.predict("Pésimo servicio")
        
        assert result == "negativo"
    
    def test_analyze_returns_complete_response(self):
        model = FakeModel(sentiment="positivo", probas=[0.85, 0.10, 0.05])
        analyzer = SentimentAnalyzer(model)
        
        result = analyzer.analyze("Excelente atención")
        
        assert result["sentiment"] == "positivo"
        assert result["score"] == 0.85
        assert "confidence" in result
        assert result["confidence"]["positivo"] == 0.85
        assert result["confidence"]["neutral"] == 0.10
        assert result["confidence"]["negativo"] == 0.05
    
    def test_analyze_score_matches_predicted_sentiment(self):
        model = FakeModel(sentiment="neutral", probas=[0.2, 0.6, 0.2])
        analyzer = SentimentAnalyzer(model)
        
        result = analyzer.analyze("El producto está bien")
        
        assert result["sentiment"] == "neutral"
        assert result["score"] == 0.6
    
    def test_predict_raises_error_on_empty_text(self):
        analyzer = SentimentAnalyzer(FakeModel())
        
        with pytest.raises(ValueError, match="texto no puede estar vacío"):
            analyzer.predict("")
    
    def test_predict_raises_error_on_whitespace_only(self):
        analyzer = SentimentAnalyzer(FakeModel())
        
        with pytest.raises(ValueError, match="texto no puede estar vacío"):
            analyzer.predict("   ")
    
    def test_predict_raises_error_on_none(self):
        analyzer = SentimentAnalyzer(FakeModel())
        
        with pytest.raises(ValueError, match="texto no puede estar vacío"):
            analyzer.predict(None)
    
    def test_analyze_raises_error_on_empty_text(self):
        analyzer = SentimentAnalyzer(FakeModel())
        
        with pytest.raises(ValueError, match="texto no puede estar vacío"):
            analyzer.analyze("")