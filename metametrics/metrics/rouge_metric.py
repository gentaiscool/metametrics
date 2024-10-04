from typing import List, Union
from .base_metric import BaseMetric
import logging

import tempfile
import os
import shutil

from pyrouge import Rouge155

class ROUGEMetric(BaseMetric):
    def __init__(self, rouge_type="rouge1", rouge_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ROUGE-1.5.5"), **kwargs):
        if not os.path.isdir(rouge_dir) and not os.path.isdir(os.environ['ROUGE_HOME']):
            logging.error("ROUGE HOME is not found")
        
        self.r = Rouge155(rouge_dir=rouge_dir, rouge_args=None, log_level=logging.ERROR)
        self.rouge_type = rouge_type

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        segment_scores = []
        
        for pred, refs in zip(predictions, references):
            self.r.system_dir = tempfile.mkdtemp()
            self.r.model_dir = tempfile.mkdtemp()
            self.r.system_filename_pattern = 'system.(\d+).txt'
            self.r.model_filename_pattern = 'model.[A-Z].#ID#.txt'
            with open(os.path.join(self.r.system_dir, "system.0.txt"), "w") as outputf:
                outputf.write(pred)
            for ref_idx, ref in enumerate(refs):
                with open(os.path.join(self.r.model_dir, f"model.{chr(ord('A') + ref_idx)}.0.txt"), "w") as outputf:
                    outputf.write(ref)
            
            output = self.r.convert_and_evaluate()
            output_dict = self.r.output_to_dict(output)
            shutil.rmtree(self.r.system_dir)
            shutil.rmtree(self.r.model_dir)
            segment_scores.append(output_dict[self.rouge_type])

        return segment_scores
        