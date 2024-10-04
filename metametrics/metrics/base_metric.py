from typing import List, Union
import torch

class BaseMetric:
    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        raise NotImplementedError()
    
class VisionToTextBaseMetric:
    def score(self, image_sources: List[torch.Tensor], text_predictions: List[str], text_references: Union[None, List[str]]=None, text_sources: Union[None, List[str]]=None) -> List[float]:
        raise NotImplementedError()