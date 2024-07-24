import pandas as pd
from typing import List, Union
from GEMBA.gemba.gpt_api import GptApi
from GEMBA.gemba.CREDENTIALS import credentials
from GEMBA.gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer

from .base_metric import BaseMetric

class GEMBA_MQM(BaseMetric):
    """
        IMPORTANT!
            Before using GEMBA_MQM, go to the GEMBA/gemba submodule and edit the CREDENTIALS.py.
            You require an API key from OpenAI to utilize this metric.
            Furthermore, to use other models, you must add entries to CREDENTIALS.py's deployments entry.
        args:
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
    def __init__(self, model: str, verbose: bool=False):
        self.model = model 
        self.verbose = verbose

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        source = [x.strip() for x in sources]
        hypothesis = [x.strip() for x in predictions] 

        assert len(source) == len(hypothesis), "Source and hypothesis list must have the same number of entries."

        df = pd.DataFrame({
            'source_seg': source,
            'target_seg': hypothesis
        })
        df['source_lang'] = "source_lang"
        df['target_lang'] = "target_lang"
        df['prompt'] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        gptapi = GptApi(credentials, verbose=self.verbose)
        answers = gptapi.bulk_request(self.df, self.model, lambda x: parse_mqm_answer(x, list_mqm_errors=False, full_desc=True), cache=None, max_token=500)
        return answers
