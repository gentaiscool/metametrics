########################## CODE ADAPTED FROM Summa-QA REPOSITORY (https://github.com/ThomasScialom/summa-qa) ########################## 

import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import spacy

from typing import List, Union
from collections import Counter
import string
import subprocess
import re

from .base_metric import BaseMetric

class QA_Bert():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.SEP_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        if torch.cuda.is_available():
            self.model.to("cuda")
    
    def predict(self, input_ids, token_type_ids, attention_mask):

        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start_scores = torch.functional.F.softmax(start_scores, -1) * token_type_ids.float()
        end_scores = torch.functional.F.softmax(end_scores, -1) * token_type_ids.float()

        start_values, start_indices = start_scores.topk(1)
        end_values, end_indices = end_scores.topk(1)

        probs = []
        asws = []
        for idx, (input_id, start_index, end_index) in enumerate(zip(input_ids, start_indices, end_indices)):
            cur_toks = self.tokenizer.convert_ids_to_tokens(input_id)
            asw = ' '.join(cur_toks[start_index[0] : end_index[0]+1])
            prob = start_values[idx][0] * end_values[idx][0]
            asws.append(asw)
            probs.append(prob.item())
        return asws, probs
        
class SummaQAMetric(BaseMetric):
    def __init__(self, model_metric="f1", batch_size=8, max_seq_len=384, use_gpu=True, **kwargs):
        self.nlp_web_sm = spacy.load("en_core_web_sm")
        self.nlp_web_md = spacy.load("en_core_web_md")
        self.model = QA_Bert()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.gpu = use_gpu
        self.model_metric = "f1" if model_metric not in ["prob", "f1"] else model_metric
    
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def get_questions(self, text_input):
        """
        Generate a list of questions on a text
        Args:
          text_input: a string
        Returns:
          a list of question
        """
        masked_questions = []
        asws = []

        for sent in text_input.sents:
            for ent in sent.ents:
                id_start = ent.start_char - sent.start_char
                id_end = ent.start_char - sent.start_char + len(ent.text)
                masked_question = sent.text[:id_start] + "MASKED" + sent.text[id_end:]
                masked_questions.append(masked_question)
                asws.append(ent.text)

        return masked_questions, asws
    
    def compute(self, questions, true_asws, evaluated_text):
        """
        Calculate the QA scores for a given text we want to evaluate and a list of questions and their answers.
        Args:
          questions: a list of string
          true_asws: a list of string
          evaluated_text: a string
        Returns:
          a dict containing the probability score and the f-score
        """
        if not questions:
            return {"avg_prob": 0, "avg_fscore": 0}

        score_prob, score_f = 0, 0
        probs = []
        asws = []
        slines = []

        for count, (question, true_asw) in enumerate(zip(questions, true_asws)):
            if count % self.batch_size == 0 and count != 0:
                input_ids = torch.tensor([ex['input_ids'] for ex in slines])
                token_type_ids = torch.tensor([ex['token_type_ids'] for ex in slines])
                attention_mask = torch.tensor([ex['attention_mask'] for ex in slines])
                if self.gpu:
                    input_ids = input_ids.to("cuda")
                    token_type_ids = token_type_ids.to("cuda")
                    attention_mask = attention_mask.to("cuda")
                asw_pred, prob = self.model.predict(input_ids, token_type_ids, attention_mask)
                asws.extend(asw_pred)
                probs.extend(prob)
                slines = []
            cur_dict = self.model.tokenizer.encode_plus(question, evaluated_text, max_length=self.max_seq_len, padding="max_length", return_token_type_ids=True)
            slines.append(cur_dict)

        if slines != []:
            input_ids = torch.tensor([ex['input_ids'] for ex in slines])
            token_type_ids = torch.tensor([ex['token_type_ids'] for ex in slines])
            attention_mask = torch.tensor([ex['attention_mask'] for ex in slines])
            if self.gpu:
                input_ids = input_ids.to("cuda")
                token_type_ids = token_type_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")
            asw_pred, prob = self.model.predict(input_ids, token_type_ids, attention_mask)
            asws.extend(asw_pred)
            probs.extend(prob)

        for asw, true_asw in zip(asws, true_asws):
            score_f += self.f1_score(asw, true_asw)
        score_prob = sum(probs)

        return {"avg_prob": score_prob/len(questions), "avg_fscore": score_f/len(questions)}
    
    
    def score(self, predictions: List[str], references: Union[None,List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        """
        Calculate the QA scores for an entire corpus.
        Args:
        sources: a list of string (one string per document)
        predictions: a list of string (one string per summary)
        a dict containing the probability score and f-score, averaged for the corpus
        """
        avg_prob_arr = []
        avg_f1_arr = []
        
        disable = ["tagger", "textcat"]
        sources = [self.nlp_web_sm(src, disable=disable) for src in sources]

        for src, pred in zip(sources, predictions):
            masked_questions, masked_question_asws = self.get_questions(src)
            pred_score = self.compute(masked_questions, masked_question_asws, pred)
            avg_prob_arr.append(pred_score['avg_prob'])
            avg_f1_arr.append(pred_score['avg_fscore'])

        if self.model_metric == "prob":
            return avg_prob_arr
        else:
            # Should be f1
            return avg_f1_arr
