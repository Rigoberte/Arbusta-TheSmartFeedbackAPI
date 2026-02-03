from abc import ABC, abstractmethod
from typing import List


class Model(ABC):
    """Interfaz para modelos de clasificación de sentimiento."""
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        """Predice el sentimiento para una lista de textos."""
        pass
    
    @abstractmethod
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        """Retorna probabilidades de cada clase para una lista de textos."""
        pass