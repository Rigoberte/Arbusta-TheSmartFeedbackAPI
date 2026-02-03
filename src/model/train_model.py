import joblib
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model.base import Model

class SklearnModel(Model):
    """Implementación de Model usando sklearn Pipeline."""
    
    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline
    
    def predict(self, texts: List[str]) -> List[str]:
        return self._pipeline.predict(texts).tolist()
    
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        return self._pipeline.predict_proba(texts).tolist()
    
    @classmethod
    def load(cls, path: Path) -> "SklearnModel":
        """Carga un modelo desde un archivo .pkl"""
        pipeline = joblib.load(path)
        return cls(pipeline)
    
    @classmethod
    def train(cls, data_path: str, output_path: str) -> "SklearnModel":
        """Entrena y guarda el modelo."""
        df = pd.read_csv(data_path)
        X, y = df['message'], df['sentiment']
        
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        pipeline.fit(X_train, y_train)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, output_path)
        
        return cls(pipeline)