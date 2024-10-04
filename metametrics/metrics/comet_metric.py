import os
from typing import List, Union
from pathlib import Path
import logging

from comet import load_from_checkpoint
from huggingface_hub import snapshot_download

from metametrics.metrics.base_metric import BaseMetric

class COMETMetric(BaseMetric):
    def __init__(self, comet_model: str="Unbabel/XCOMET-XXL", batch_size: int=8, gpus: int=1,
                 reference_free: bool=False, hf_token: str="", **kwargs):
        # Choose model from the model hub
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.batch_size = batch_size
        self.gpus = gpus
        self.reference_free = reference_free
        self.hf_token = hf_token if hf_token != "" else os.environ['HF_TOKEN']
        if self.hf_token == "":
            logging.warning("HuggingFace Token is not filled, this may cause")
            
        model_path = self.download_model(comet_model, saving_directory=os.path.join(cur_dir, f"comet_models/{comet_model}"))
        self.model = load_from_checkpoint(model_path)

    # Copy from comet's download_model to include hf_token
    def download_model(self, model: str, saving_directory: Union[str, Path, None]=None,
                       local_files_only: bool=False) -> str:
        try:
            model_path = snapshot_download(
                repo_id=model, cache_dir=saving_directory, local_files_only=local_files_only, token=self.hf_token,
            )
        except Exception:
            raise KeyError(f"COMET failed to download model '{model}'.")
        else:
            checkpoint_path = os.path.join(*[model_path, "checkpoints", "model.ckpt"])
        return checkpoint_path

    def score(self, predictions: List[str], references: Union[None, List[str]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        # Data must be converted into the following format for COMET
        if self.reference_free:
            data = [{"src": src, "mt": mt} for src, mt in zip(sources, predictions)]
        else:
            data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(sources, predictions, references)]
        return self.model.predict(data, batch_size=self.batch_size, gpus=self.gpus).scores
