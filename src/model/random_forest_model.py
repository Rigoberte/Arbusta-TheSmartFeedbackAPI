import joblib
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.model.base import Model


class RandomForestModel(Model):
    """
    Implementación de Model usando Random Forest.
    """
    
    def __init__(self, pipeline: Pipeline):
        self._pipeline: Pipeline = pipeline
    
    def predict(self, texts: List[str]) -> List[str]:
        return self._pipeline.predict(texts).tolist()
    
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        return self._pipeline.predict_proba(texts).tolist()
    
    @classmethod
    def load(cls, path: Path) -> "RandomForestModel":
        """Carga un modelo desde un archivo .pkl"""
        pipeline = joblib.load(path)
        return cls(pipeline)
    
    @classmethod
    def train(cls, data_path: str, output_path: str) -> "RandomForestModel":
        """Entrena y guarda el modelo Random Forest."""
        df = pd.read_csv(data_path)
        X, y = df['message'], df['sentiment']
        
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(
                n_estimators=100,      # número de árboles
                max_depth=20,          # profundidad máxima
                random_state=42,
                n_jobs=-1             # usa todos los cores
            ))
        ])
        pipeline.fit(X_train, y_train)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, output_path)
        
        return cls(pipeline)