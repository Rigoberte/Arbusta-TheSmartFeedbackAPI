import pytest
from pathlib import Path

from src.model.random_forest_model import RandomForestModel
from src.model.base import Model


class TestRandomForestModelIntegration:
    @pytest.fixture
    def trained_model(self, tmp_path: Path) -> RandomForestModel:
        """Entrena un modelo temporal para tests."""
        data_path = Path("data/reviews.csv")
        if not data_path.exists():
            pytest.skip("reviews.csv no encontrado")
        
        output_path = tmp_path / "test_model.pkl"
        return RandomForestModel.train(str(data_path), str(output_path))
    
    def test_train_creates_model_file(self, tmp_path):
        data_path = Path("data/reviews.csv")
        if not data_path.exists():
            pytest.skip("reviews.csv no encontrado")
        
        output_path = tmp_path / "test_model.pkl"
        
        RandomForestModel.train(str(data_path), str(output_path))
        
        assert output_path.exists()
    
    def test_load_returns_random_forest_model(self, tmp_path):
        data_path = Path("data/reviews.csv")
        if not data_path.exists():
            pytest.skip("reviews.csv no encontrado")
        
        output_path = tmp_path / "test_model.pkl"
        RandomForestModel.train(str(data_path), str(output_path))
        
        loaded = RandomForestModel.load(output_path)
        
        assert isinstance(loaded, RandomForestModel)
    
    def test_predict_returns_valid_sentiment(self, trained_model: Model):
        result = trained_model.predict(["Me encanta este producto"])
        
        assert len(result) == 1
        assert result[0] in ["positivo", "neutral", "negativo"]
    
    def test_predict_handles_multiple_texts(self, trained_model: Model):
        texts = ["Excelente", "Malo", "Normal"]
        
        result = trained_model.predict(texts)
        
        assert len(result) == 3
        assert all(s in ["positivo", "neutral", "negativo"] for s in result)
    
    def test_predict_proba_returns_three_probabilities(self, trained_model: Model):
        result = trained_model.predict_proba(["Buen producto"])
        
        assert len(result) == 1
        assert len(result[0]) == 3  # 3 clases
    
    def test_predict_proba_sums_to_one(self, trained_model: Model):
        result = trained_model.predict_proba(["Texto de prueba"])
        
        total = sum(result[0])
        assert abs(total - 1.0) < 0.01
    
    def test_predict_proba_all_positive_values(self, trained_model: Model):
        result = trained_model.predict_proba(["Cualquier texto"])
        
        assert all(p >= 0 for p in result[0])
