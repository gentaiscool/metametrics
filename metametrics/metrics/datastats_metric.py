######################## ADAPTED FROM https://github.com/Yale-LILY/SummEval/blob/master/evaluation/summ_eval/data_stats_metric.py ########################

from evaluate import load
from typing import List, Union
from .base_metric import BaseMetric

from collections import Counter, namedtuple
import spacy
import re

# Taken from https://github.com/lil-lab/newsroom/blob/master/newsroom/analyze/fragments.py

class Fragments:
    Match = namedtuple("Match", ("summary", "text", "length"))
    def __init__(self, summary, text, case=False):
        if isinstance(summary, str):
            self.summary = summary.split()
        else:
            self.summary = summary
        if isinstance(text, str):
            self.text = text.split()
        else:
            self.text = text

        self._norm_summary = self.normalize(self.summary, case)
        self._norm_text = self.normalize(self.text, case)

        self._match(self._norm_summary, self._norm_text)
        
    def normalize(self, tokens, case=False):
        """
        Lowercases and turns tokens into distinct words.
        """
        return [str(t).lower() if not case else str(t) \
            for t in tokens]

    def density(self, summary_base=True):
        """
        Return the DENSITY score of summary and text.
        Arguments:
            - summary_base (bool): use summary as numerator (default = True)
        Returns:
            - decimal DENSITY score within [0, ...]
        """
        numerator = sum(o.length ** 2 for o in self._matches)
        if summary_base:
            denominator = len(self.summary)
        else:
            denominator = len(self.text)

        if denominator == 0:
            return 0
        else:
            return numerator / denominator
        
    def get_percentage_repeated_n_gram(self, n_gram):       
        summ_ngrams = list(zip(*[self._norm_summary[i:] for i in range(n_gram)]))
        summ_ngrams_set = set(summ_ngrams)
        ngramCounter = Counter()
        ngramCounter.update(summ_ngrams)
        repeated = [key for key, val in ngramCounter.items() if val > 1]
        if len(summ_ngrams_set) != 0:
            return len(repeated) / float(len(summ_ngrams_set))
        else:
            return 0

    def _match(self, a, b):
        """
        Raw procedure for matching summary in text, described in paper.
        """
        self._matches = []
        a_start = b_start = 0
        while a_start < len(a):
            best_match = None
            best_match_length = 0

            while b_start < len(b):
                if a[a_start] == b[b_start]:
                    a_end = a_start
                    b_end = b_start

                    while a_end < len(a) and b_end < len(b) and b[b_end] == a[a_end]:
                        b_end += 1
                        a_end += 1

                    length = a_end - a_start
                    if length > best_match_length:
                        best_match = Fragments.Match(a_start, b_start, length)
                        best_match_length = length
                    b_start = b_end
                else:
                    b_start += 1

            b_start = 0
            if best_match:
                if best_match_length > 0:
                    self._matches.append(best_match)
                a_start += best_match_length
            else:
                a_start += 1

class DataStatsMetric(BaseMetric):
    def __init__(self, stats_type="density", case=False, tokenize=False, **kwargs):
        self.nlp_web_sm = spacy.load('en_core_web_sm')
        self.nlp_web_md = spacy.load('en_core_web_md')
        if stats_type == "density":
            self.stats_type = stats_type
        elif re.fullmatch(r'^repeated_\d+-gram$', stats_type) != 0:
            self.stats_type = stats_type
        else:
            self.stats_type = "density"
        self.case = case
        self.tokenize = tokenize

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        disable = ["tagger", "textcat", "ner"]
        sources = [self.nlp_web_sm(src, disable=disable) for src in sources]
        sources = [[tok.text for tok in src] for src in sources]
        predictions = [self.nlp_web_sm(pred, disable=disable) for pred in predictions]
        predictions = [[tok.text for tok in pred] for pred in predictions]
        
        scores = []
        
        for input_text, summary in zip(sources, predictions):
            fragments = Fragments(summary, input_text, case=self.case)
            
            if self.stats_type == "density":
                scores.append(fragments.density())
            else:
                # We only work with repeated n-gram types
                n_gram = int(re.findall(r'repeated_(\d+)-gram', self.stats_type)[0])
                scores.append(fragments.get_percentage_repeated_n_gram(n_gram))
                
        return scores
