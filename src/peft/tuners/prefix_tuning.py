# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
from dataclasses import dataclass, field
from typing import Optional, Union
import torch
import torch.nn as nn

from ..utils import PeftType, PromptLearningConfig

class PrefixTuningInit(str, enum.Enum):
    MLP = "MLP"
    TEXT = "TEXT"
    RANDOM = "RANDOM"

@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.PrefixEncoder`].

    Args:
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        prefix_projection (`bool`): Whether to project the prefix embeddings.
    """

    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    
    prefix_projection: str = field(
        default="MLP",
        metadata={"help": "How to project the prefix tokens: (MLP, TEXT, RANDOM)"},
    )
    
    prefix_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prefix tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer used for prefix tuning initialization. Only used if prefix_tuning_init is `TEXT`"
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model used for prefix tuning initialization. Only used if prefix_tuning_init is `TEXT`"
        },
    )
    def __post_init__(self):
        self.peft_type = PeftType.PREFIX_TUNING


# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
class PrefixEncoder(torch.nn.Module):
    r"""
    The torch.nn model to encode the prefix

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example::

        >>> from peft import PrefixEncoder, PrefixTuningConfig >>> config = PrefixTuningConfig(
                peft_type="PREFIX_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=20, token_dim=768,
                num_transformer_submodules=1, num_attention_heads=12, num_layers=12, encoder_hidden_size=768,
                prefix_projection='TEXT', prefix_tuning_init_text='',
                prefix_tuning_init_text=None, tokenizer_name_or_path='gpt2', model_name_or_path='gpt2',
            )
        >>> prefix_encoder = PrefixEncoder(config)


    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) --
            The embedding layer of the prefix encoder.
        - **transform** (`torch.nn.Sequential`) -- The
        two-layer MLP to transform the prefix embeddings if `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (batch_size, num_virtual_tokens)

    Output shape: (batch_size, num_virtual_tokens, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        self.num_virtual_tokens = config.num_virtual_tokens
        if self.prefix_projection=="RANDOM" or config.inference_mode:
            self.embedding = torch.nn.Embedding(self.num_virtual_tokens, num_layers * 2 * token_dim)
        elif self.prefix_projection=="MLP":
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(self.num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
        elif self.prefix_projection=="TEXT":
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer_name_or_path = config.tokenizer_name_or_path
            self.model_name_or_path = config.model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
            self.model = AutoModel.from_pretrained(self.model_name_or_path)
            self.prefix_tuning_init_text = config.prefix_tuning_init_text
            # self.model = config.model
            self.embedding = nn.Parameter(self._get_gold_init())
        else:
            raise ValueError("self.prefix_projection is not supported")

    def _get_gold_init(self):
        init_text = self.prefix_tuning_init_text
        init_token_ids = self.tokenizer(init_text, return_tensors='pt')["input_ids"]
        # Trim or iterate until num_text_tokens matches total_virtual_tokens
        self.num_virtual_tokens = len(init_token_ids)
        
        self.model = self.model.cuda()
        with torch.no_grad():
            try:
                output = self.model(init_token_ids.to(self.model.device), return_dict=True, use_cache=True)
            except:
                print('init_token_ids: ', init_token_ids)
                print('init_token_ids.shape: ', init_token_ids.shape)
            output = output.past_key_values
            output = output.past_key_values
            init_val = []
            for item in output:
                init_val.append(item[0].unsqueeze(0)) # key, [1, 1, num_heads, sequence_length, dim_head]
                init_val.append(item[1].unsqueeze(0)) # val
            output = torch.cat(init_val, dim=0)
            print("=== Sanity Check ===")
            print('init past_key_values: ', output.shape)
            # print("init past_key_value for each layer as: ", len(output), len(output[0]), output[0][0].shape)
            # output = torch.cat(output, dim=0)
        return output
        
    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
