import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Union
from tqdm import tqdm

class YiSiModel(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased', idf_weights: Dict[int, float] = None):
        super(YiSiModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.idf_weights = idf_weights

    def get_token_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        return embeddings

    def compute_cosine_similarity_matrix(self, embeddings1, embeddings2):
        norm1 = embeddings1.norm(dim=2, keepdim=True)
        norm2 = embeddings2.norm(dim=2, keepdim=True)
        cos_sim_matrix = torch.bmm(embeddings1, embeddings2.transpose(1, 2)) / (norm1 * norm2.transpose(1, 2))
        return cos_sim_matrix

    def compute_weighted_pool(self, similarities, input_ids):
        idf_weights = torch.FloatTensor([[self.idf_weights.get(tok.item(), 1.0) for tok in seq] for seq in input_ids]).to(input_ids.device)
        weighted_sum = torch.bmm(similarities.unsqueeze(1), idf_weights.unsqueeze(2)).squeeze(-1)
        total_weight = idf_weights.sum(dim=1, keepdim=True)
        return weighted_sum / total_weight

    def forward(self, pred_input_ids, pred_attention_mask, ref_input_ids, ref_attention_mask):
        # Get embeddings
        pred_embeddings = self.get_token_embeddings(pred_input_ids, pred_attention_mask)
        ref_embeddings = self.get_token_embeddings(ref_input_ids, ref_attention_mask)

        # Compute cosine similarity matrix
        cos_sim_matrix = self.compute_cosine_similarity_matrix(pred_embeddings, ref_embeddings)

        # Get maximum similarity for each token in prediction and reference sentences
        max_similarities_pred, _ = cos_sim_matrix.max(dim=2)
        max_similarities_ref, _ = cos_sim_matrix.max(dim=1)

        # Compute weighted pool for prediction and reference tokens
        weighted_pool_pred = self.compute_weighted_pool(max_similarities_pred, pred_input_ids)
        weighted_pool_ref = self.compute_weighted_pool(max_similarities_ref, ref_input_ids)

        return weighted_pool_pred, weighted_pool_ref

class YiSiMetric:
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', alpha: float = 0.8, batch_size=64, max_input_length=512, device='cuda'):
        self.model_name = model_name
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = YiSiModel(model_name=model_name).to(self.device)

    def _compute_idf(self, documents: List[str]) -> Dict[str, float]:
        tf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 1),
            tokenizer=self.tokenizer.tokenize,
            token_pattern=None
        ).fit(documents)

        return {self.tokenizer.convert_tokens_to_ids([tok])[0]: tf.idf_[tf.vocabulary_[tok]] for tok in tf.vocabulary_.keys()}

    def tokenize(self, texts: List[str]):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=self.max_input_length).to(self.device)

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        idf_weights = self._compute_idf(predictions + references)
        self.model.idf_weights = idf_weights

        scores = []

        for start_idx in tqdm(range(0, len(predictions), self.batch_size), desc="Scoring", ncols=100):
            end_idx = min(start_idx + self.batch_size, len(predictions))
            batch_predictions = predictions[start_idx:end_idx]
            batch_references = references[start_idx:end_idx]

            pred_inputs = self.tokenize(batch_predictions)
            ref_inputs = self.tokenize(batch_references)

            weighted_pool_pred, weighted_pool_ref = self.model(
                pred_input_ids=pred_inputs['input_ids'],
                pred_attention_mask=pred_inputs['attention_mask'],
                ref_input_ids=ref_inputs['input_ids'],
                ref_attention_mask=ref_inputs['attention_mask']
            )

            for precision, recall in zip(weighted_pool_pred, weighted_pool_ref):
                precision = precision.item()
                recall = recall.item()
                score = (precision * recall) / (self.alpha * precision + (1 - self.alpha) * recall)
                scores.append(score)

        return scores
