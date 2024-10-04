#################### ADAPTED FROM https://github.com/Yale-LILY/SummEval/blob/master/evaluation/summ_eval/rouge_we_metric.py ####################

from evaluate import load
from typing import List, Union
from .base_metric import BaseMetric

import collections
import six
import os

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import numpy as np
from scipy import spatial

class ROUGEWEMetric(BaseMetric):
    def __init__(self, n_gram=1, model_metric="f1", tokenize=False, **kwargs):
        self.n_gram = n_gram
        self.model_metric = "f1" if model_metric not in ["precision", "recall", "f1"] else model_metric
        self.tokenize = tokenize
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopset = frozenset(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.word_embeddings = self.load_embeddings(os.path.join(cur_dir, "embeddings/deps.words"))
        self.alpha = 0.5
        self.threshold = 0.8

    ###################################################
    ###             Pre-Processing
    ###################################################

    def pre_process_summary_stem(self, summary, stem=True):
        all_words = []
        if self.tokenize:
            for s in summary:
                if stem:
                    all_words.extend([self.stemmer.stem(r) for r in self.tokenizer.tokenize(s)])
                else:
                    all_words.extend(self.tokenizer.tokenize(s))
        else:
            if isinstance(summary, list):
                all_words = summary[0].split()
            else:
                all_words = summary.split()

        normalized_content_words = [word.lower() for word in all_words]
        return normalized_content_words

    def _ngrams(self, words, n):
        queue = collections.deque(maxlen=n)
        for w in words:
            queue.append(w)
            if len(queue) == n:
                yield tuple(queue)

    def _ngram_counts(self, words, n):
        return collections.Counter(self._ngrams(words, n))

    def _ngram_count(self, words, n):
        return max(len(words) - n + 1, 0)

    def calculate_model_metric(self, matches, recall_total, precision_total):
        precision_score = matches / precision_total if precision_total > 0 else 0.0
        recall_score = matches / recall_total if recall_total > 0 else 0.0
        denom = (1.0 - self.alpha) * precision_score + self.alpha * recall_score
        f1_score = (precision_score * recall_score) / denom if denom > 0.0 else 0.0
        
        if self.model_metric == "precision":
            return precision_score
        elif self.model_metric == "recall":
            return recall_score
        else:
            return f1_score

    def _has_embedding(self, ngram):
        for w in ngram:
            if w not in self.word_embeddings:
                return False
        return True

    def _get_embedding(self, ngram):
        res = []
        for w in ngram:
            res.append(self.word_embeddings[w])
        return np.sum(np.array(res), 0)

    def _find_closest(self, ngram, counter):
        ## If there is nothin to match, nothing is matched
        if len(counter) == 0:
            return "", 0, 0

        ## If we do not have embedding for it, we try lexical matching
        if not self._has_embedding(ngram):
            if ngram in counter:
                return ngram, counter[ngram], 1
            else:
                return "", 0, 0

        ranking_list = []
        ngram_emb = self._get_embedding(ngram)
        for k, v in six.iteritems(counter):
            ## First check if there is an exact match
            if k == ngram:
                ranking_list.append((k, v, 1.))
                continue

            ## if no exact match and no embeddings: no match
            if not self._has_embedding(k):
                ranking_list.append((k, v, 0.))
                continue

            ## soft matching based on embeddings similarity
            k_emb = self._get_embedding(k)
            ranking_list.append((k, v, 1 - spatial.distance.cosine(k_emb, ngram_emb)))

        ## Sort ranking list according to sim
        ranked_list = sorted(ranking_list, key=lambda tup: tup[2], reverse=True)

        ## extract top item
        return ranked_list[0]

    def _soft_overlap(self, peer_counter, model_counter):
        result = 0
        for k, v in six.iteritems(peer_counter):
            closest, count, sim = self._find_closest(k, model_counter)
            if sim < self.threshold:
                continue
            if count <= v:
                del model_counter[closest]
                result += count
            else:
                model_counter[closest] -= v
                result += v

        return result
    
    def load_embeddings(self, filepath):
        dict_embedding = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.rstrip().split(" ")
                key = line[0]
                vector = line[1::]
                dict_embedding[key.lower()] = np.array([float(x) for x in vector])
        return dict_embedding

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:        
        segment_scores = []
        
        for pred, refs in zip(predictions, references):
            if len(refs) == 1 and isinstance(refs[0], str):
                refs = [refs]
            pred = self.pre_process_summary_stem(pred, False)
            refs = [self.pre_process_summary_stem(ref, False) for ref in refs]

            matches = 0
            recall_total = 0
            pred_counter = self._ngram_counts(pred, self.n_gram)
            for ref in refs:
                model_counter = self._ngram_counts(ref, self.n_gram)
                matches += self._soft_overlap(pred_counter, model_counter)
                recall_total += self._ngram_count(ref, self.n_gram)
            precision_total = len(refs) * self._ngram_count(pred, self.n_gram)
            segment_scores.append(self.calculate_model_metric(matches, recall_total, precision_total))

        return segment_scores