from typing import List, Union
from .base_metric import BaseMetric
import sacrebleu

# none, 13a, intl, char, spm, flores101, flores200

class BLEUMetric(BaseMetric):
    def __init__(self, smooth_method="exp", smooth_value=None,
                 use_effective_order=True, tokenize='13a',
                 lowercase=True, **kwargs):
        self.smooth_method = smooth_method
        self.smooth_value = smooth_value
        self.use_effective_order = use_effective_order
        self.tokenize = tokenize
        self.lowercase = lowercase

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        segment_scores = []
        for pred, ref in zip(predictions, references):
            score = sacrebleu.sentence_bleu(pred, ref, lowercase=self.lowercase, tokenize=self.tokenize,
                                            smooth_method=self.smooth_method, smooth_value=self.smooth_value,
                                            use_effective_order=self.use_effective_order).score
            segment_scores.append(score)
        return segment_scores