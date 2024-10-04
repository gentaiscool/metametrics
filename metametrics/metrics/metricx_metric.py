from metametrics.metrics.base_metric import BaseMetric
import json
from torch.utils.data import Dataset
import torch
from torch import nn
import transformers
import transformers.modeling_outputs

from typing import Dict, List, Union
import datasets
import os
import torch
import transformers
import uuid

import copy
import dataclasses
from typing import Optional, Tuple, Union
import warnings

BaseModelOutput = transformers.modeling_outputs.BaseModelOutput
ModelOutput = transformers.modeling_outputs.ModelOutput

MT5Config = transformers.models.mt5.modeling_mt5.MT5Config
MT5PreTrainedModel = transformers.models.mt5.modeling_mt5.MT5PreTrainedModel
MT5Stack = transformers.models.mt5.modeling_mt5.MT5Stack

__HEAD_MASK_WARNING_MSG = (
    transformers.models.mt5.modeling_mt5.__HEAD_MASK_WARNING_MSG  # pylint: disable=protected-access
)


@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
  loss: Optional[torch.FloatTensor] = None
  predictions: torch.FloatTensor = None


class MT5ForRegression(MT5PreTrainedModel):
  """MT5 model for regression."""

  def __init__(self, config: MT5Config):
    super().__init__(config)
    self.model_dim = config.d_model

    self.shared = nn.Embedding(config.vocab_size, config.d_model)

    encoder_config = copy.deepcopy(config)
    encoder_config.is_decoder = False
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    self.encoder = MT5Stack(encoder_config, self.shared)

    decoder_config = copy.deepcopy(config)
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    decoder_config.num_layers = config.num_decoder_layers
    self.decoder = MT5Stack(decoder_config, self.shared)

    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    # Initialize weights and apply final processing
    self.post_init()

    # Model parallel
    self.model_parallel = False
    self.device_map = None

  def forward(
      self,
      input_ids: Optional[torch.LongTensor] = None,
      attention_mask: Optional[torch.FloatTensor] = None,
      decoder_attention_mask: Optional[torch.BoolTensor] = None,
      head_mask: Optional[torch.FloatTensor] = None,
      decoder_head_mask: Optional[torch.FloatTensor] = None,
      cross_attn_head_mask: Optional[torch.Tensor] = None,
      encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
      labels: Optional[torch.FloatTensor] = None,
      use_cache: Optional[bool] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
  ) -> Union[Tuple[torch.FloatTensor], MT5ForRegressionOutput]:
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # FutureWarning: head_mask was separated into two input args - head_mask,
    # decoder_head_mask
    if head_mask is not None and decoder_head_mask is None:
      if self.config.num_layers == self.config.num_decoder_layers:
        warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        decoder_head_mask = head_mask

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
      # Convert encoder inputs in embeddings if needed
      encoder_outputs = self.encoder(
          input_ids=input_ids,
          attention_mask=attention_mask,
          inputs_embeds=inputs_embeds,
          head_mask=head_mask,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
      )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
      encoder_outputs = BaseModelOutput(
          last_hidden_state=encoder_outputs[0],
          hidden_states=encoder_outputs[1]
          if len(encoder_outputs) > 1
          else None,
          attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
      )

    hidden_states = encoder_outputs[0]

    if self.model_parallel:
      torch.cuda.set_device(self.decoder.first_device)

    # Create 1 step of dummy input for the decoder.
    batch_size = input_ids.size(0)
    decoder_input_ids = torch.LongTensor([0]).repeat(batch_size).reshape(-1, 1)
    if torch.cuda.is_available():
      decoder_input_ids = decoder_input_ids.to(torch.device("cuda"))

    # Set device for model parallelism
    if self.model_parallel:
      torch.cuda.set_device(self.decoder.first_device)
      hidden_states = hidden_states.to(self.decoder.first_device)
      if decoder_input_ids is not None:
        decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
      if attention_mask is not None:
        attention_mask = attention_mask.to(self.decoder.first_device)
      if decoder_attention_mask is not None:
        decoder_attention_mask = decoder_attention_mask.to(
            self.decoder.first_device
        )

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
      torch.cuda.set_device(self.encoder.first_device)
      self.lm_head = self.lm_head.to(self.encoder.first_device)
      sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
      # Rescale output before projecting on vocab
      # See
      # https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
      sequence_output = sequence_output * (self.model_dim**-0.5)

    lm_logits = self.lm_head(sequence_output)

    # 250089 = <extra_id_10>
    predictions = lm_logits[:, 0, 250089]

    # Clip to 0 to 25
    predictions = torch.clamp(predictions, 0, 25)

    loss = None
    if labels is not None:
      loss_fct = nn.MSELoss()
      # move labels to correct device to enable PP
      labels = labels.to(predictions.device)
      loss = loss_fct(predictions.view(-1), labels.view(-1))

    return MT5ForRegressionOutput(
        loss=loss,
        predictions=predictions,
    )

class MetricXMetric(BaseMetric):
    def __init__(self, is_qe: bool, tokenizer_name: str, model_name: str, batch_size: int,
                 max_input_length: int, bf16: bool, **kwargs):
        self.reference_free = is_qe
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.bf16 = bf16
        if self.bf16:
            self.model = MT5ForRegression.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        else:
            self.model = MT5ForRegression.from_pretrained(model_name)

        self.max_input_length = max_input_length

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.per_device_batch_size = batch_size // torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.per_device_batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

        self.training_args = transformers.TrainingArguments(
            per_device_eval_batch_size=self.per_device_batch_size,
            dataloader_pin_memory=False,
            output_dir=".",
            bf16=self.bf16,
        )
        if self.per_device_batch_size == 1:
            self.trainer = transformers.Trainer(
                model=self.model,
                args=self.training_args,
            )
        else:
            data_collator = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
            self.trainer = transformers.Trainer(
                model=self.model,
                args=self.training_args,
                data_collator = data_collator
            )

    def get_dataset(self, sources:Union[List[str], None], hypothesis:List[str], references:List[str]):
        """Gets the test dataset for prediction.

        If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
        If it is false, there must be "hypothesis" and "reference" fields.

        Args:
            sources: a list of sources
            hypothesis: a list of hypothesis
            references: a list of gold references
            max_input_length: The maximum input sequence length.
            device: The ID of the device to put the PyTorch tensors on.
            is_qe: Indicates whether the metric is a QE metric or not.

        Returns:
            The dataset.
        """

        def _make_input(example):
            if self.reference_free:
                example["input"] = (
                    "candidate: "
                    + example["hypothesis"]
                    + " source: "
                    + example["source"]
                )
            else:
                example["input"] = (
                    "candidate: "
                    + example["hypothesis"]
                    + " reference: "
                    + example["reference"]
                )
            return example

        def _tokenize(example):
            if self.per_device_batch_size == 1:
                return self.tokenizer(
                    example["input"],
                    max_length=self.max_input_length,
                    truncation=True,
                    padding=False,
                )
            else:
                return self.tokenizer(
                    example["input"],
                    max_length=self.max_input_length,
                    truncation=True,
                    # padding=False,
                    padding='max_length',
                )

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

        data_obj = []
        for i in range(len(hypothesis)):
            new_obj = {}
            if sources is not None:
                new_obj["source"] = sources[i]
            if hypothesis is not None:
                new_obj["hypothesis"] = hypothesis[i]
            if references is not None:
                new_obj["reference"] = references[i]
            data_obj.append(new_obj)

        input_file = str(uuid.uuid1()) + ".json"
        print(f"input file: {input_file}")
        with open(input_file, "w+") as f:
            f.write(json.dumps(data_obj) + "\n")

        ds = datasets.load_dataset("json", data_files={"test": input_file})
        ds = ds.map(_make_input)
        ds = ds.map(_tokenize)
        ds = ds.map(_remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.device,
            output_all_columns=True,
        )
        os.system(f"rm {input_file}")
        return ds

    def score(self, predictions:List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[List[str]]]=None) -> List[float]:
        ds = self.get_dataset(
            sources,
            predictions,
            references,
        )
 
        predictions, _, _ = self.trainer.predict(test_dataset=ds["test"])
        return predictions
