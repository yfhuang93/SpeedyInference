# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch.nn.functional as F
import torch
import transformers

from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from confidence_measures import compute_confidence, should_exit

@dataclass
class ForwardResult:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    exit_query_cache: Optional[List[torch.Tensor]] = None
    hidden_states: Optional[List[torch.Tensor]] = None

# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(model, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = transformers.generation.logits_process.TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = transformers.generation.logits_process.TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits

def decode_next_token(
    logits: torch.Tensor,
    token_idx: int = None,
    sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
) -> torch.Tensor:
    if token_idx:
        logits = logits[:, -1, :]

    if not sample:
        next_token = logits.argmax(dim=-1)
        return next_token, None
    else:
        if not token_idx:
            logits.squeeze_(dim=0)
        filtered_logits = top_k_top_p_filtering(logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        if not token_idx:
            next_token.transpose_(1, 0)
        return next_token, probabilities


def crop_past_key_values(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    maximum_length: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(len(past_key_values)):
        if past_key_values[idx] is None or past_key_values[idx][0] == [] or past_key_values[idx][0] is None:
            break
        new_past.append(
            (
                past_key_values[idx][0][:, :, :maximum_length, :],
                past_key_values[idx][1][:, :, :maximum_length, :],
            )
        )
    past_key_values = tuple(new_past)
    return past_key_values


# Our forward_early(...) and forward_remainder(...) functions currently use transformers library's legacy KV cache implementation that is less efficient.
# To ensure an apples to apples comparison, we created this forward function to use in autoregressive decoding to ensure it uses the same KV cache implementation instead.
# FIXME: update forward_early(...) and forward_remainder(...) to use the updated more efficient KV cache implementation.
def forward(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )
    all_hidden_states = []
    hidden_states = inputs_embeds
    for decoder_layer in model.model.layers:
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )
    all_hidden_states.append(hidden_states.clone())

    past_key_values = past_key_values.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return ForwardResult(
        logits=logits, past_key_values=past_key_values,hidden_states=all_hidden_states
    )


# TODO: update forward_early(...) to use transformers' new KV cache implementation rather than legacy.
def forward_early(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    all_hidden_states = []
    for decoder_layer in model.model.layers[:exit_layer]:
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )
    all_hidden_states.append(hidden_states.clone())

    past_key_values = past_key_values.to_legacy_cache()

    # next_cache = next_decoder_cache
    if exit_query_cache is None:
        exit_query_cache = hidden_states
    else:
        exit_query_cache = torch.cat([exit_query_cache, hidden_states], dim=1)

    hidden_states = model.model.norm(hidden_states)

    logits = model.lm_head(hidden_states)
    return ForwardResult(
        logits=logits, past_key_values=past_key_values, exit_query_cache=exit_query_cache,hidden_states=all_hidden_states
    )



# TODO: update forward_remainder(...) to use transformers' new KV cache implementation rather than legacy.
def forward_remainder(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape
    num_tokens_to_generate: int = 1
    seq_length_with_past = seq_length
    draft_past_key_values_length: int = 0
    full_past_key_values_length: int = 0

    # Print input information
    # print(f"Input tensor shape: {input_ids.shape}")

    if past_key_values is not None and past_key_values[0] is not None:
        # it's okay to use the first layer because the draft model necessairly computes it
        draft_past_key_values_length = past_key_values[0][0].shape[2]
        # the total sequence length is the past key values since that includes the draft tokens

        # the last layer should not have been skipped, we can get this to check how many of the tokens have gone through full
        # verification
        if len(past_key_values) == len(model.model.layers):
            full_past_key_values_length = past_key_values[-1][0].shape[2]
        else:
            # we have not done a full pass yet so the history is 0
            full_past_key_values_length = 0

        seq_length_with_past = num_tokens_to_generate + draft_past_key_values_length

    # # Print past_key_values details
    # print(f"draft_past_key_values_length: {draft_past_key_values_length}")
    # print(f"full_past_key_values_length: {full_past_key_values_length}")

    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    inputs_embeds = model.model.embed_tokens(input_ids)

    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length_with_past,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    early_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, num_tokens_to_generate),
        inputs_embeds,
        draft_past_key_values_length,
    )

    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,  # we have no past for the full model
    )

    next_decoder_cache = []
    hidden_states = inputs_embeds
    all_hidden_states = []
    # TODO simplify
    full_hidden_states: Optional[torch.FloatTensor] = None
    for idx, decoder_layer in enumerate(model.model.layers):
        is_early_exit = idx < exit_layer

        # print(f"Processing layer {idx + 1}/{len(model.model.layers)}. Early exit: {is_early_exit}")

        past_key_value = (
            past_key_values[idx]
            if (past_key_values is not None and idx < len(past_key_values))
            else None
        )
        if is_early_exit:
            # early hidden states: B x num_gen x C
            early_hidden_states = hidden_states[:, -num_tokens_to_generate:]
            early_position_ids = position_ids[:, -num_tokens_to_generate:]

            # print(f"attention_mask shape: {attention_mask.shape}")
            # if full_hidden_states is not None:
            #     print(f"full_hidden_states shape: {full_hidden_states.shape}")
            # print(
            #     f"past_key_values shapes: {[f'layer {idx}: {[t.shape for t in p]}' for idx, p in enumerate(past_key_values)]}")

            hidden_states, past_key_values = decoder_layer(
                early_hidden_states,
                attention_mask=early_attention_mask,
                position_ids=early_position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )
        else:
            if full_hidden_states is None and exit_query_cache is not None:

                # print(f"Using exit_query_cache for layer {idx + 1}")
                # print(f"exit_query_cache shape: {exit_query_cache.shape}")
                # print(f"hidden_states shape before concat: {hidden_states[:, -num_tokens_to_generate:].shape}")

                # first time seeing the full hidden states, we need to rely on the
                # query cache
                # only use if exit query cache exists, if not this is our first call
                full_hidden_states = torch.cat(
                    [exit_query_cache, hidden_states[:, -num_tokens_to_generate:]],
                    dim=1,
                )
                # print(f"full_hidden_states shape after concat: {full_hidden_states.shape}")
            else:
                # we already have seen the fully hidden states we can re-use them now
                full_hidden_states = hidden_states
            # try:
            hidden_states, past_key_values = decoder_layer(
                full_hidden_states,
                attention_mask=full_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
                )
            # except RuntimeError as e:
            #     print(f"Error in Layer {idx + 1}")
            #     print(f"full_hidden_states shape: {full_hidden_states.shape}")
            #     print(f"attention_mask shape: {full_attention_mask.shape}")
            #     print(f"position_ids shape: {position_ids.shape}")
            #     print(f"past_key_value shapes: {[p.shape for p in past_key_values]}")
            #     raise e

    # # Print hidden state and past key value details
    # print(f"Hidden state shape after layer {idx + 1}: {hidden_states.shape}")
    # if past_key_value is not None:
    #     print(
    #         f"past_key_values shapes: {[f'layer {idx}: {[t.shape for t in p]}' for idx, p in enumerate(past_key_values)]}")

    all_hidden_states.append(hidden_states.clone())

    past_key_values = past_key_values.to_legacy_cache()

    # Print exit query cache details
    # if exit_query_cache is not None:
    #     print(f"exit_query_cache shape before final layer: {exit_query_cache.shape}")

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    # print(f"Final logits shape: {logits.shape}")

    return ForwardResult(
        logits=logits, past_key_values=past_key_values, exit_query_cache=exit_query_cache,hidden_states=all_hidden_states
    )

def forward_early_with_CALM(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
    position: int,
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    # print(f"[forward_early_with_CALM] Input tensor shape: {input_ids.shape}")


    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
        # print(f"[forward_early_with_CALM] past_key_values length: {len(past_key_values)}")
        # print(f"past_key_values shapes: {[f'layer {idx}: {[t.shape for t in p]}' for idx, p in enumerate(past_key_values)]}")

    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)


    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    # print(f"[forward_early_with_CALM] Target exit_layer: {exit_layer}")
    # print(f"[forward_early_with_CALM] Position: {position}")

    hidden_states = inputs_embeds
    prev_state = hidden_states.clone()
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        # print(f"[forward_early_with_CALM] Processing layer {layer_idx + 1}")
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )
        # print(f"[forward_early_with_CALM] Hidden states shape after layer {layer_idx + 1}: {hidden_states.shape}")

        current_logits = model.lm_head(hidden_states)
        last_token_logits = current_logits[:, -1, :]
        new_state = hidden_states.clone()


        confidence = compute_confidence(
            logits=last_token_logits.float(),
            prev_state=prev_state[:, -1, :].float(),  # [batch_size, hidden_size]
            new_state=new_state[:, -1, :].float(),  # [batch_size, hidden_size]
            conf_method=GenerationConfig.conf_method,
        )  # [batch_size]

        # print(f"[forward_early_with_CALM] Layer {layer_idx + 1} confidence: {confidence}")

        if GenerationConfig.position_adjusted_threshold:
            adjusted_threshold = (
                GenerationConfig.conf_threshold
                * torch.exp(-GenerationConfig.position_temp * position / GenerationConfig.max_steps)
                / 10
                + 9 * GenerationConfig.conf_threshold / 10
            )
        else:
            adjusted_threshold = GenerationConfig.conf_threshold

        # print(f"[forward_early_with_CALM] Adjusted threshold: {adjusted_threshold}")

        # Decide whether to exit
        exit_now = should_exit(confidence, adjusted_threshold)
        # print(f"[forward_early_with_CALM] Exit now: {exit_now}")
        # satify confidence or minimal exit layer
        if (exit_now |((layer_idx+2)==exit_layer)):
            # print(f"[forward_early_with_CALM] Exiting at layer {layer_idx + 1}")
            break
        prev_state = new_state.clone()


    past_key_values = past_key_values.to_legacy_cache()

    # next_cache = next_decoder_cache
    if exit_query_cache is None:
        exit_query_cache = hidden_states
        # print(f"[forward_early_with_CALM] Initializing exit_query_cache with hidden_states")
    else:
        # print(f"[forward_early_with_CALM] exit_query_cache shape before concat: {exit_query_cache.shape}")
        # print(f"[forward_early_with_CALM] hidden_states[:, -seq_length:] shape: {hidden_states[:, -seq_length:].shape}")
        exit_query_cache = torch.cat([exit_query_cache, hidden_states], dim=1)
        # exit_query_cache = torch.cat([exit_query_cache, hidden_states[:, -seq_length:]], dim=1)
        # print(f"[forward_early_with_CALM] exit_query_cache shape after concat: {exit_query_cache.shape}")

    hidden_states = model.model.norm(hidden_states)

    logits = model.lm_head(hidden_states)

    # print(f"[forward_early_with_CALM] Final logits shape: {logits.shape}")
    # print(f"[forward_early_with_CALM] Final exit_layer set to: {layer_idx + 2}")

    GenerationConfig.final_exit_layer=layer_idx+2
    return ForwardResult(
        logits=logits, past_key_values=past_key_values, exit_query_cache=exit_query_cache
    )
