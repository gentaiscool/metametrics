import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Union, Optional

from metametrics.metrics.base_metric import TextBaseMetric
from metametrics.utils.validate import validate_bool
from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

class ArmoRMMetric(TextBaseMetric):
    """
    ArmoRM Metric for evaluating text responses based on multiple scoring attributes.

    Args:
        model_id (str): The model identifier for loading ArmoRM.
        scoring_attributes (List[str]): A list of attributes used for scoring.
        trust_remote_code (bool): Whether to trust remote code when loading the model.

    Example usecase:
    metric = ArmoRMMetric(scoring_attributes=["helpsteer-helpfulness", "ultrafeedback-truthfulness"])
    scores = metric.score(predictions=preds, sources=questions)

    # Example output
    # [
    #     {"helpsteer-helpfulness": 0.85, "ultrafeedback-truthfulness": 0.92},
    #     {"helpsteer-helpfulness": 0.67, "ultrafeedback-truthfulness": 0.81},
    #     ...
    # ]

    """

    class ArmoRMPipeline:
        def __init__(self, model_id: str, trust_remote_code: bool = False):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id, device_map="auto", trust_remote_code=trust_remote_code, torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            self.device = self.model.device
            self.truncation = True
            self.max_length = 4096
            self.attributes = [
                'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
                'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
                'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
                'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
                'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm', 'code-complexity',
                'code-style', 'code-explanation', 'code-instruction-following', 'code-readability'
            ]
            self.attribute_dict = {k: v for (v, k) in enumerate(self.attributes)}

        def __call__(self, messages: List[Dict[str, str]], scoring_attributes: List[str]) -> Dict[str, float]:
            """
            Score a given set of chat messages.

            Args:
                messages (List[Dict[str, str]]): The chat conversation.
                scoring_attributes (List[str]): The scoring attributes.

            Returns:
                Dict[str, float]: The computed scores for each attribute.
            """
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", padding=True, truncation=self.truncation, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                output = self.model(input_ids)
                multi_obj_rewards = output.rewards.cpu().float()
                gating_output = output.gating_output.cpu().float()
                preference_score = output.score.cpu().float()

            obj_transform = self.model.reward_transform_matrix.data.cpu().float()
            multi_obj_coeffs = gating_output @ obj_transform.T

            assert torch.isclose(torch.sum(multi_obj_rewards * multi_obj_coeffs, dim=1), preference_score, atol=1e-3)
            
            return {attr: multi_obj_rewards[0][self.attribute_dict[attr]].item() for attr in scoring_attributes}

    def __init__(self, model_id: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1", scoring_attributes: Optional[List[str]] = None, trust_remote_code: bool = True, **kwargs):
        self.model_id = model_id
        self.trust_remote_code = validate_bool(trust_remote_code)

        self.rm = self.ArmoRMPipeline(model_id=self.model_id, trust_remote_code=self.trust_remote_code)
        if scoring_attributes is None:
            self.scoring_attributes = [self.rm.attributes[1]]  # Default to ['helpsteer-correctness']
        else:
            # Validate all provided attributes exist
            for attr in scoring_attributes:
                if attr not in self.rm.attributes:
                    raise ValueError(f"Invalid scoring attribute: {attr}")
            self.scoring_attributes = scoring_attributes

    def score(self, predictions: List[str], references: Union[None, List[List[str]]] = None, sources: Union[None, List[str]] = None) -> List[Dict[str, float]]:
        if sources is None or predictions is None:
            raise ValueError("Both sources (questions) and predictions (answers) must be provided.")

        df = pd.DataFrame({"question": sources, "prediction": predictions})
        scores = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                score = self.rm(
                    messages=[{"role": "user", "content": row['question']}, {"role": "assistant", "content": row['prediction']}],
                    scoring_attributes=self.scoring_attributes
                )
                scores.append(score)
            except Exception as e:
                logger.error(f"Error computing score: {e}. Assigning default score of -99 for all attributes.")
                raise RuntimeError(f"Failed to compute scores due to: {e}")

        return scores

    @property
    def min_val(self) -> Optional[float]:
        return -1.0 # Note to David: I'm not sure what min_val affects, I'm using -99 as an error code. Is this okay?

    @property
    def max_val(self) -> Optional[float]:
        return 1.0

    @property
    def higher_is_better(self) -> bool:
        """Indicates if a higher value is better for this metric."""
        return True

    def __eq__(self, other):
        if isinstance(other, ArmoRMMetric):
            self_vars = {k: v for k, v in vars(self).items() if k != 'rm'}
            other_vars = {k: v for k, v in vars(other).items() if k != 'rm'}
            return self_vars == other_vars

        return False
