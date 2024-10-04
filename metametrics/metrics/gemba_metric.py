import pandas as pd
import json
from typing import List, Union
from metametrics.metrics.GEMBA.gemba.gpt_api import GptApi
from metametrics.metrics.GEMBA.gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer
from metametrics.metrics.base_metric import BaseMetric

class GEMBA_MQM(BaseMetric):
    """
        IMPORTANT!
            Before using GEMBA_MQM, go to the GEMBA/gemba submodule and edit the CREDENTIALS.py.
            You require an API key from OpenAI to utilize this metric.
            Furthermore, to use other models, you must add entries to CREDENTIALS.py's deployments entry.
        args:
            credentials (dict): Your credentials should follow the following pattern
                credentials = {
                    "deployments": {"text-davinci-002": "text-davinci-002"},
                    "api_key": "********************************",
                    "requests_per_second_limit": 1
                }
            verbose (bool): defaults to False, set to True to test output results.

        GEMBA_MQM is a reference-free metric.
        A large-language model is used to automatically check for errors in model translation (hypothesis) from the source.
        Errors are separated into 3 categories: Critical, Major, and Minor.
            each critical errors have a weight of -25
            each major errors have a weight of -5
            each minor errors have a weight of -1
            GEMBA_MQM's score ranges from -25 to 0, with 0 meaning no errors detected.

        Example Usage:
        gemba_metric = GEMBA_MQM('gpt-4')
        source = ["I like pie"]
        hypothesis = ["Saya suka pie"]
        gemba_score = gemba_metric.score(
            source_lang="English",
            target_lang="Indonesian",
            source=source,
            hypothesis=hypothesis
        )
    """
    def __init__(self, model: str, credentials: dict, source_lang: str, target_lang: str, verbose: bool=False):
        self.model = model 
        self.verbose = verbose
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        if "deployments" not in credentials:
            raise ValueError('"deployments" key not found in credentials')
        if "api_key" not in credentials:
            raise ValueError('"api_key" key not found in credentials')
        if "requests_per_second_limit" not in credentials:
            raise ValueError('"requests_per_second_limit" key not found in credentials')
            
        self.credentials = credentials

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        source = [x.strip() for x in sources]
        hypothesis = [x.strip() for x in predictions] 

        assert len(source) == len(hypothesis), "Source and hypothesis list must have the same number of entries."

        df = pd.DataFrame({
            'source_seg': source,
            'target_seg': hypothesis
        })
        df['source_lang'] = self.source_lang
        df['target_lang'] = self.target_lang
        df['prompt'] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        gptapi = GptApi(self.credentials, verbose=self.verbose)
        gptapi.non_batchable_models += [self.model]

        answers = gptapi.bulk_request(df, self.model, lambda x: parse_mqm_answer(x, list_mqm_errors=False, full_desc=True), cache=None, max_tokens=500)
        with open("gemba_output.json", 'w') as f:
            json.dump(answers, f)
        answers_list = [x['answer'] for x in answers]
        return answers_list
    
