# Copyright 2024 Hao Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"

class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        print("=" * 50)
        print("DEBUG: LlavaQwenModel.__init__() started")
        
        # Cek struktur config
        if hasattr(config, 'text_config') and config.text_config is not None:
            text_config = config.text_config

            print("DEBUG: Calling Qwen2Model.__init__() with text_config...")
            Qwen2Model.__init__(self, text_config)
        else:
            print("DEBUG: Calling Qwen2Model.__init__() with direct config...")
            Qwen2Model.__init__(self, config)
        
        print("DEBUG: Calling LlavaMetaModel.__init__()...")
        LlavaMetaModel.__init__(self, config)
        
        print("DEBUG: LlavaQwenModel.__init__() finished successfully")
        print("=" * 50)

@dataclass
class LlavaOutputWithPast(CausalLMOutputWithPast):
    labels: Optional[torch.FloatTensor] = None

class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        print("=" * 50)
        print("DEBUG: LlavaQwenForCausalLM.__init__() started")
        
        config_attrs = [attr for attr in dir(config) if not attr.startswith('_')]
        print(f"DEBUG: Config attributes: {config_attrs}")
        
        # Cek atribut penting
        important_attrs = ['vocab_size', 'hidden_size', 'text_config', 'vision_config']
        for attr in important_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"DEBUG: config.{attr} = {value} (type: {type(value)})")
            else:
                print(f"DEBUG: config.{attr} = MISSING")
        
        # Handle config detection dengan lebih robust
        if hasattr(config, 'text_config') and config.text_config is not None:
            print("DEBUG: Detected config with text_config structure")
            text_config = config.text_config
            # print(f"DEBUG: text_config type: {type(text_config)}")
            # print(f"DEBUG: text_config.vocab_size = {getattr(text_config, 'vocab_size', 'MISSING')}")
            # print(f"DEBUG: text_config.hidden_size = {getattr(text_config, 'hidden_size', 'MISSING')}")
            
            full_config = config
            
            Qwen2ForCausalLM.__init__(self, text_config)
            
            self.config = full_config
            print("DEBUG: Set self.config to full_config")
            
        else:
            print("DEBUG: Detected direct config (no text_config)")
            text_config = config
            # print(f"DEBUG: Direct config vocab_size = {getattr(config, 'vocab_size', 'MISSING')}")
            # print(f"DEBUG: Direct config hidden_size = {getattr(config, 'hidden_size', 'MISSING')}")
            
            Qwen2ForCausalLM.__init__(self, config)
            
            self.config = config
            print("DEBUG: Set self.config to direct config")
        
        # Initialize model multimodal
        self.model = LlavaQwenModel(self.config)
        print("DEBUG: LlavaQwenModel initialized successfully")
        
        self.lm_head = nn.Linear(
            text_config.hidden_size,
            text_config.vocab_size, 
            bias=False
        )
        print("DEBUG: lm_head re-initialized")
        
        # Pastikan weights di-initialize dengan benar
        self.post_init()
        print("DEBUG: post_init() completed")
        
        print("DEBUG: LlavaQwenForCausalLM.__init__() finished successfully")
        print("=" * 50)
        
    def _init_weights(self, module):
        """
        Override _init_weights untuk handle config LlavaOnevisionConfig
        """
        print("=" * 50)
        print("DEBUG: _init_weights called")
        print(f"DEBUG: Module type: {type(module).__name__}")
        
        # Simpan config asli
        original_config = self.config
        
        try:
            # Gunakan text_config untuk weight initialization jika ada
            if hasattr(self.config, 'text_config') and self.config.text_config is not None:
                print("DEBUG: Using text_config for weight initialization")
                self.config = self.config.text_config
                # print(f"DEBUG: text_config.initializer_range: {getattr(self.config, 'initializer_range', 'MISSING')}")
            else:
                print("DEBUG: Using direct config for weight initialization")
                # print(f"DEBUG: direct config.initializer_range: {getattr(self.config, 'initializer_range', 'MISSING')}")
            
            # Jika masih tidak ada initializer_range, set default value
            if not hasattr(self.config, 'initializer_range'):
                print("DEBUG: initializer_range missing, setting default value 0.02")
                self.config.initializer_range = 0.02
            
            # print(f"DEBUG: Final config.initializer_range: {self.config.initializer_range}")
            
            # Panggil parent _init_weights
            super()._init_weights(module)
            print("DEBUG: Parent _init_weights completed successfully")
            
        except Exception as e:
            print(f"DEBUG: ERROR in _init_weights: {e}")
            print(f"DEBUG: Error type: {type(e).__name__}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            raise
        finally:
            # Pastikan config asli selalu dikembalikan
            self.config = original_config
            print(f"DEBUG: Finished init weights")
            print("=" * 50)
            
    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        print("\n")
        print("=" * 50)
        print("DEBUG: LlavaQwenForCausalLM.forward() started")
        
        if input_ids is not None: print(f"DEBUG: Initial input_ids shape: {input_ids.shape}")
        if images is not None:
            if isinstance(images, list): shapes = [str(im[0].shape) for im in images]; print(f"DEBUG: Initial images: list of {len(shapes)} items with shapes like {shapes[0]}")
            else: print(f"DEBUG: Initial images shape: {images.shape}")
        if labels is not None: print(f"DEBUG: Initial labels shape: {labels.shape}")

        if inputs_embeds is None:
            print("DEBUG: inputs_embeds is None. Calling prepare_inputs_labels_for_multimodal...")
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
            if inputs_embeds is not None: print(f"DEBUG: After prepare_inputs, new inputs_embeds shape: {inputs_embeds.shape}")
            if labels is not None: print(f"DEBUG: After prepare_inputs, new labels shape: {labels.shape}")
        else:
            print(f"DEBUG: inputs_embeds is already provided. Shape: {inputs_embeds.shape}")

        original_config = self.config
        output = None
        try:
            # print("DEBUG: Temporarily swapping self.config to text_config before calling parent forward...")
            self.config = original_config.text_config
            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            print("DEBUG: Parent forward() call completed successfully.")
        finally:
            self.config = original_config
            
        if output is not None: print(f"DEBUG: Final output type: {type(output)}")
        else: print("DEBUG: Final output is None.")
        print("DEBUG: LlavaQwenForCausalLM.forward() finished")
        print("=" * 50)
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)