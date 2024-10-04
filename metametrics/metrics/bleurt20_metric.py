import os
from typing import List, Union
import numpy as np

from bleurt import score

from metametrics.metrics.base_metric import BaseMetric

class BLEURT20Metric(BaseMetric):
    def __init__(self, agg_method: str="mean", **kwargs):
        # The paths
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        bleurt_model_path = os.path.join(cur_dir, "bleurt/BLEURT-20")     
        self.scorer = score.BleurtScorer(bleurt_model_path)
        self.agg_method = agg_method if agg_method in ["mean", "max"] else "mean"

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        # Note the references can be different for each prediction, so we should handle this case
        max_ref_length = max(len(ref_list) for ref_list in references)
        
        # Initialize an array to store scores, filled with NaN initially
        all_scores = np.full((len(predictions), max_ref_length), np.nan)
        
        # Iterate over each reference index (up to max_ref_length)
        for ref_idx in range(max_ref_length):
            # Collect the valid indices where references exist
            valid_indices = [i for i in range(len(references)) if ref_idx < len(references[i])]
            
            # Collect references and predictions for valid indices
            current_refs = [references[i][ref_idx] for i in valid_indices]
            current_preds = [predictions[i] for i in valid_indices]
            
            # Compute BLEURT scores for valid references
            if current_refs:
                ref_scores = self.scorer.score(references=current_refs, candidates=current_preds)
                
                # Fill the corresponding positions in the all_scores array
                for idx, score in zip(valid_indices, ref_scores):
                    all_scores[idx, ref_idx] = score

        # Aggregate the scores using the chosen method, ignoring NaN values
        if self.agg_method == 'mean':
            aggregated_scores = np.nanmean(all_scores, axis=1)
        elif self.agg_method == 'max':
            aggregated_scores = np.nanmax(all_scores, axis=1)
        
        return aggregated_scores
    
