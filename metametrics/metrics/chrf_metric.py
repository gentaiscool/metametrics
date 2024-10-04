from evaluate import load
from typing import List, Union
from .base_metric import BaseMetric

class chrFMetric(BaseMetric):
    def __init__(self, word_order=2, eps_smoothing=True, **kwargs):
        self.hf_metric = load("chrf")
        self.word_order = word_order # word_order == 2 means chrF++
        self.eps_smoothing = eps_smoothing  # eps_smoothing means chrF++

    def score(self, predictions: List[str], references: Union[None,List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        segment_scores = []
        for pred, ref in zip(predictions, references):
            score = self.hf_metric.compute(predictions=[pred], references=[ref], word_order=self.word_order, eps_smoothing=self.eps_smoothing)['score']
            segment_scores.append(score)
        return segment_scores