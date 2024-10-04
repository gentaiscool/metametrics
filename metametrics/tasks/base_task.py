from typing import List, Union
from abc import ABC, abstractmethod

class BaseTask(ABC):
    @abstractmethod
    def add_metric(self, metric):
        raise NotImplementedError()
    
    @abstractmethod
    def set_optimizer(self, optimizer):
        raise NotImplementedError()
    