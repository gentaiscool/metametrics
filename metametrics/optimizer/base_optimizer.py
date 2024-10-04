from typing import List, Union
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(self, X_train, Y_train, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, X_feats):
        raise NotImplementedError()
    