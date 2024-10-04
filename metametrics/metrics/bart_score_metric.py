from evaluate import load
from typing import List, Union
from metametrics.metrics.BARTScore.bart_score import BARTScorer
from metametrics.metrics.base_metric import BaseMetric

class BARTScoreMetric(BaseMetric):
    """
        args:
            metric_args (Dict): a dictionary of metric arguments
    """
    def __init__(self, model_checkpoint: str='facebook/bart-large-cnn', max_length: int=1024,
                 agg_method: str="max", batch_size: int=8, use_gpu=True, **kwargs):
        self.batch_size = batch_size
        self.agg_method = agg_method if agg_method in ["mean", "max"] else "mean"
        if use_gpu:
            self.bart_scorer = BARTScorer(device='cuda:0', max_length=max_length, checkpoint=model_checkpoint)
        else:
            self.bart_scorer = BARTScorer(device='cpu', max_length=max_length, checkpoint=model_checkpoint)

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[List[str]]]=None) -> List[float]:
        all_scores = self.bart_scorer.multi_ref_score(predictions, references, agg=self.agg_method, batch_size=self.batch_size)
        return all_scores