# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import List, Optional, Tuple

import torch

import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import (
    decode_next_token,
    forward,
    forward_early,
)

import torch.nn.functional as F
from confidence_measures import compute_confidence, should_exit


class AutoRegressiveGenerationStrategy(GenerationStrategy):
    def generate_token_ids(
            self,
            model: transformers.LlamaForCausalLM,
            input_ids: List[int],
            eos_token_id: int,
            generation_config: GenerationConfig,
            logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
            stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
            streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationStrategyResult:
        """Variant of `generate` with inputs/outputs formatted as token_ids."""
        past_key_values = None

        input_ids: torch.Tensor = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []

        exit_query_cache = None
        for _ in range(generation_config.max_steps):
            if generation_config.exit_layer > 0:
                model_output = forward_early(
                    model,
                    input_ids,
                    past_key_values,
                    generation_config.exit_layer,
                    exit_query_cache,
                )
            else:
                model_output = forward(
                    model,
                    input_ids,
                    past_key_values,
                )
            logits = model_output.logits
            if logits_processors:
                logits = logits_processors(input_ids, logits)
            past_key_values = model_output.past_key_values
            next_token, _ = decode_next_token(logits=logits, token_idx=-1, sample=generation_config.sample,
                                              temperature=generation_config.temperature, top_k=generation_config.top_k,
                                              top_p=generation_config.top_p)
            if streamer:
                streamer.put(next_token)
            next_token = next_token.item()
            if next_token == eos_token_id:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
            output_ids.append(next_token)
            # Don't concatenate `next_token` to original `input_ids` since we're using
            # the KV cache (`past_key_values`) to speed up generation.
            input_ids = torch.tensor([[next_token]]).to(input_ids)

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,
        )


class AutoRegressiveGenerationStrategyWithCALM(GenerationStrategy):
    def generate_token_ids(
            self,
            model: transformers.LlamaForCausalLM,
            input_ids: List[int],
            eos_token_id: int,
            generation_config: GenerationConfig,
            logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
            stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
            streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationStrategyResult:

        past_key_values = None
        input_ids: torch.Tensor = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []
        exit_query_cache = None

        batch_size = input_ids.size(0)
        prev_hidden_state = torch.zeros(batch_size, model.config.hidden_size).to(input_ids.device)

        exit_layer = generation_config.exit_layer
        min_exit_layer = generation_config.min_exit_layer
        middle_layer = exit_layer // 2 if exit_layer // 2 > min_exit_layer else min_exit_layer + 1
        critical_layers = [min_exit_layer, middle_layer]

        for step in range(generation_config.max_steps):
            token_exit_layer = exit_layer

            for layer in critical_layers:
                model_output = forward_early(
                    model,
                    input_ids,
                    past_key_values,
                    layer,
                    exit_query_cache,
                )

                logits = model_output.logits
                if logits_processors:
                    logits = logits_processors(input_ids, logits)
                past_key_values = model_output.past_key_values
                last_token_logits = logits[:, -1, :]

                if model_output.hidden_states is not None and len(model_output.hidden_states) > 0:
                    new_state = model_output.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
                else:
                    new_state = torch.zeros(batch_size, model.config.hidden_size).to(input_ids.device)

                confidence = compute_confidence(
                    logits=last_token_logits,
                    prev_state=prev_hidden_state,
                    new_state=new_state,
                    conf_method=generation_config.conf_method,
                )

                exit_now = should_exit(confidence, generation_config.conf_threshold)
                prev_hidden_state = new_state

                if exit_now.any():
                    # print("Current_layer:",layer)
                    token_exit_layer = layer
                    break

            model_output = forward_early(
                    model,
                    input_ids,
                    past_key_values,
                    token_exit_layer,
                    exit_query_cache,
            )

            logits = model_output.logits
            if logits_processors:
                logits = logits_processors(input_ids, logits)
            past_key_values = model_output.past_key_values
            next_token, _ = decode_next_token(logits=logits, token_idx=-1, sample=generation_config.sample,
                                              temperature=generation_config.temperature, top_k=generation_config.top_k,
                                              top_p=generation_config.top_p)
            if streamer:
                streamer.put(next_token)
            next_token = next_token.item()
            if next_token == eos_token_id:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
            output_ids.append(next_token)
            input_ids = torch.tensor([[next_token]]).to(input_ids)

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,
        )




