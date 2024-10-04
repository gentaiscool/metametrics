from evaluate import load
from typing import List, Union
from metametrics.metrics.base_metric import BaseMetric

class BERTScoreMetric(BaseMetric):
    """
        args:
            metric_args (Dict): a dictionary of metric arguments
    """
    def __init__(self, model_name: str, model_metric: str, num_layers: int=None,
                 batch_size: int=8, nthreads: int=16, rescale_with_baseline: bool=False, **kwargs):
        self.hf_metric = load("bertscore")
        self.model_name = model_name
        self.model_metric = "f1" if model_metric not in ["precision", "recall", "f1"] else model_metric
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.nthreads = nthreads
        self.rescale_with_baseline = rescale_with_baseline

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[List[str]]]=None) -> List[float]:
        all_scores = self.hf_metric.compute(predictions=predictions, references=references, 
                                          model_type=self.model_name, batch_size=self.batch_size,
                                          nthreads=self.nthreads, num_layers=self.num_layers,
                                          lang="en", rescale_with_baseline=self.rescale_with_baseline)[self.model_metric]
        return all_scores
