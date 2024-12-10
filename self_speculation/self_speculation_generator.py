# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time

from typing import List, Optional, Tuple

import colorama
import torch

import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategySkip,
    GenerationStrategyResult,
)
from self_speculation.speculative_streamer import SpeculativeTextStreamer
from self_speculation.llama_model_utils import (
    crop_past_key_values,
    decode_next_token,
    forward_early,
    forward_remainder,
    forward_early_with_CALM,
    LlamaForCausalLM,
)

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

class SelfSpeculativeGenerationStrategy(GenerationStrategy):
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

        input_ids_list = input_ids
        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)
        output_ids: List[int] = []

        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        while len(output_ids) < generation_config.max_steps:
            (
                input_ids,
                output_ids,
                past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation(
                model=model,
                input_ids_list=input_ids_list,
                input_ids=input_ids,
                output_ids=output_ids,
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                exit_layer=generation_config.exit_layer,
                eos_token_id=eos_token_id,
                calls=calls,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += num_speculations
            eos_found = False
            if eos_token_id in output_ids:
                # break out of loop when we get an EOS token
                # remove the EOS token id
                output_ids = output_ids[: output_ids.index(eos_token_id)]
                eos_found = True
            if eos_found:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=total_draft_matches / total_generations,
        )
    
    # TODO: remove calls, input_ids_list, rely on generation config
    def single_step_speculation(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_id: int,
        calls: int,
        exit_layer: int,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
        th_stop_draft: Optional[float]=0.7,
        draft_exiting: Optional[bool]=True,
    ):
        time1 = time.time()
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        exit_query_cache = None
        for _ in range(num_speculations):
            # time1 = time.time()
            draft_result = forward_early(
                model,
                draft_input_ids,
                past_key_values,
                exit_layer,
                exit_query_cache,
            )
            # time2 = time.time()
            # print(time2-time1)
            # past_key_values = draft_result.past_key_values
            # exit_query_cache = draft_result.exit_query_cache
            draft_logits = draft_result.logits
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)
            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            # print(draft_next_prob.shape,draft_next_token.shape)
            if draft_exiting:# and draft_next_prob.item() < th_stop_draft:
                draft_output_probs = torch.gather(draft_next_prob, -1, draft_next_token).squeeze(-1)
                if draft_output_probs.item() < th_stop_draft and draft_output_ids!=[]:# and (1-random_list[step_draft]) <= th_random_draft) or step + step_draft + 2 >= max_new_tokens:                
                    break                   
            past_key_values = draft_result.past_key_values
            # print("~~~~~~~",type(draft_result.past_key_values),"~~~~~~~~")
            exit_query_cache = draft_result.exit_query_cache
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            # print(draft_output_ids)
            if sample:
                draft_probabilities.append(draft_next_prob)
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            if draft_next_token == eos_token_id:
                # break out of loop when we get an EOS token
                break

        # input_ids (1 x T_p) and draft_output_ids (1 x T_d) are concatenated together to make
        # 1 x (T_d  + T_p)
        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
        # print(draft_output_ids.shape)
        prefill_token_ids = torch.cat(
            [input_ids, draft_output_ids],
            dim=-1,
        )
        # print(prefill_token_ids.int().shape)

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                print(colorama.Fore.LIGHTMAGENTA_EX, end="")
                streamer.put(draft_output_ids, is_draft=True)
                
        time2 = time.time()
        # print(time2-time1)

        # logits: 1 x (T_d  + T_p) x V
        verify_results = forward_remainder(
            model,
            prefill_token_ids.int(),
            past_key_values,
            exit_layer,
            exit_query_cache,
        )
        logits = verify_results.logits
        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        past_key_values = verify_results.past_key_values
        # only select the logits relevant to what the draft has outputted.
        # verification_logits: 1 x T_d x V
        verification_logits = logits[:, prompt_length - 1 :, :]

        # verified_tokens: 1 x (T_d)
        # There is a predicted token for every token in the draft output ids list, however note that the
        # first tokens (or first N tokens) are coming from the prompt
        verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)

        # skip verification of the last token as it is a new token predicted from the main model
        verified_tokens = verified_tokens.to(prefill_token_ids)
        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]

        # number of matches is the index of the number of tokens we are accepting from the draft
        if not sample:
            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids, dtype=torch.float)
            for i in range(draft_output_ids.numel()):
                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[0, i]].item() / draft_probabilities[i][0, draft_output_ids[0, i]].item()):
                    number_of_matches += 1
                else:
                    verified_tokens[0][number_of_matches] = torch.multinomial(max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()
                    break

        # accept the `number_of_matches` tokens from the draft with one more from the main model
        # since we re-use the same cachem the input id should only be the last accepted token TODO check this
        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]
        output_ids.extend(draft_output_ids[0, : number_of_matches].tolist())
        output_ids.extend(verified_tokens[0][number_of_matches : number_of_matches + 1].tolist())

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                streamer.delete(len(draft_output_ids[0, :]))
                print(colorama.Fore.GREEN, end="")
                streamer.put(draft_output_ids[0, : number_of_matches])
                print(colorama.Style.RESET_ALL, end="")
                streamer.put(verified_tokens[0][number_of_matches : number_of_matches + 1])
            else:
                # streamer.put(torch.cat((draft_output_ids[0, : number_of_matches], verified_tokens[0][number_of_matches : number_of_matches + 1])))
                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))

        # we want the entire output sequence + input sequence
        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids_list) + len(output_ids) - 1
        )
        
        time3 = time.time()
        # print("-",time3-time2)

        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            draft_output_ids.numel(),
        )
    
    
    
class SelfSpeculativeGenerationStrategyWithCALM(GenerationStrategy):
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

        input_ids_list = input_ids
        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)
        output_ids: List[int] = []

        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        ## use single_step_speculation to generate multiple steps; the result are used to decided whether to stop;
        while len(output_ids) < generation_config.max_steps:
            (
                input_ids,
                output_ids,
                past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation(
                model=model,
                input_ids_list=input_ids_list,
                input_ids=input_ids,
                output_ids=output_ids,
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                exit_layer=generation_config.exit_layer,
                eos_token_id=eos_token_id,
                calls=calls,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += num_speculations
            eos_found = False
            if eos_token_id in output_ids:
                # break out of loop when we get an EOS token
                # remove the EOS token id
                output_ids = output_ids[: output_ids.index(eos_token_id)]
                eos_found = True
            if eos_found:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=total_draft_matches / total_generations,
        )

    # TODO: remove calls, input_ids_list, rely on generation config
    def single_step_speculation(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_id: int,
        calls: int,
        exit_layer: int,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
        th_stop_draft: Optional[float]=0.7,
        draft_exiting: Optional[bool]=True,
    ):
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        exit_query_cache = None
        for step in range(num_speculations):
            time1 = time.time()
            draft_result = forward_early_with_CALM(
                model,
                draft_input_ids,
                past_key_values,
                exit_layer,
                exit_query_cache,
                step
            )
            time2 = time.time()
            # print(time2-time1)
            # past_key_values = draft_result.past_key_values
            # exit_query_cache = draft_result.exit_query_cache
            draft_logits = draft_result.logits
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)
            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            # print(draft_next_prob.shape,draft_next_token.shape)
            if draft_exiting:# and draft_next_prob.item() < th_stop_draft:
                draft_output_probs = torch.gather(draft_next_prob, -1, draft_next_token).squeeze(-1)
                if draft_output_probs.item() < th_stop_draft and draft_output_ids!=[]:# and (1-random_list[step_draft]) <= th_random_draft) or step + step_draft + 2 >= max_new_tokens:                
                    break                   
            past_key_values = draft_result.past_key_values
            # print("~~~~~~~",type(draft_result.past_key_values),"~~~~~~~~")
            exit_query_cache = draft_result.exit_query_cache
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            # print(draft_output_ids)
            if sample:
                draft_probabilities.append(draft_next_prob)
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            if draft_next_token == eos_token_id:
                # break out of loop when we get an EOS token
                break
            # past_key_values = draft_result.past_key_values
            # exit_query_cache = draft_result.exit_query_cache
            # draft_logits = draft_result.logits
            # if logits_processors:
            #     draft_logits = logits_processors(draft_input_ids, draft_logits)
            # draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            # draft_next_token = draft_next_token.item()
            # draft_output_ids.append(draft_next_token)
            # if sample:
            #     draft_probabilities.append(draft_next_prob)
            # draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            # if draft_next_token == eos_token_id:
            #     # break out of loop when we get an EOS token
            #     break

        # input_ids (1 x T_p) and draft_output_ids (1 x T_d) are concatenated together to make
        # 1 x (T_d  + T_p)
        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
        prefill_token_ids = torch.cat(
            [input_ids, draft_output_ids],
            dim=-1,
        )

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                print(colorama.Fore.LIGHTMAGENTA_EX, end="")
                streamer.put(draft_output_ids, is_draft=True)

        # logits: 1 x (T_d  + T_p) x V
        # print("GenerationConfig.final_exit_layer",GenerationConfig.final_exit_layer)
        verify_results = forward_remainder(
            model,
            prefill_token_ids.int(),
            past_key_values,
            exit_layer, # new exit layer
            exit_query_cache,
        )
        logits = verify_results.logits
        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        past_key_values = verify_results.past_key_values
        # only select the logits relevant to what the draft has outputted.
        # verification_logits: 1 x T_d x V
        verification_logits = logits[:, prompt_length - 1 :, :]

        # verified_tokens: 1 x (T_d)
        # There is a predicted token for every token in the draft output ids list, however note that the
        # first tokens (or first N tokens) are coming from the prompt
        verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)

        # skip verification of the last token as it is a new token predicted from the main model
        verified_tokens = verified_tokens.to(prefill_token_ids)
        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]

        # number of matches is the index of the number of tokens we are accepting from the draft
        if not sample:
            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids, dtype=torch.float)
            for i in range(draft_output_ids.numel()):
                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[0, i]].item() / draft_probabilities[i][0, draft_output_ids[0, i]].item()):
                    number_of_matches += 1
                else:
                    verified_tokens[0][number_of_matches] = torch.multinomial(max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()
                    break

        # accept the `number_of_matches` tokens from the draft with one more from the main model
        # since we re-use the same cachem the input id should only be the last accepted token TODO check this
        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]
        output_ids.extend(draft_output_ids[0, : number_of_matches].tolist())
        output_ids.extend(verified_tokens[0][number_of_matches : number_of_matches + 1].tolist())

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                streamer.delete(len(draft_output_ids[0, :]))
                print(colorama.Fore.GREEN, end="")
                streamer.put(draft_output_ids[0, : number_of_matches])
                print(colorama.Style.RESET_ALL, end="")
                streamer.put(verified_tokens[0][number_of_matches : number_of_matches + 1])
            else:
                # streamer.put(torch.cat((draft_output_ids[0, : number_of_matches], verified_tokens[0][number_of_matches : number_of_matches + 1])))
                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))

        # we want the entire output sequence + input sequence
        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids_list) + len(output_ids) - 1
        )

        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            draft_output_ids.numel(),
        )
    
    
    
class SelfSpeculativeGenerationStrategySkip(GenerationStrategy):
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

        input_ids_list = input_ids
        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)
        output_ids: List[int] = []

        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        while len(output_ids) < generation_config.max_steps:
            (
                input_ids,
                output_ids,
                past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation(
                model=model,
                input_ids_list=input_ids_list,
                input_ids=input_ids,
                output_ids=output_ids,
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                exit_layer=generation_config.exit_layer,
                eos_token_id=eos_token_id,
                calls=calls,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += num_speculations
            eos_found = False
            if eos_token_id in output_ids:
                # break out of loop when we get an EOS token
                # remove the EOS token id
                output_ids = output_ids[: output_ids.index(eos_token_id)]
                eos_found = True
            if eos_found:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=total_draft_matches / total_generations,
        )
    
    # TODO: remove calls, input_ids_list, rely on generation config
    def single_step_speculation(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_id: int,
        calls: int,
        exit_layer: int,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
        th_stop_draft: Optional[float]=0.8,
        draft_exiting: Optional[bool]=False,
        draft_attn_skip_mask: torch.Tensor = None,
        draft_mlp_skip_mask: torch.Tensor = None,
    ):
        time1 = time.time()
        prompt_length: int = input_ids.size(1)
        past_key_values_predraft = past_key_values
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        exit_query_cache = None
        # local_model_path = "facebook/layerskip-llama2-7B"
        # model = LlamaForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16).to('cuda:0').eval()
        # print(type(model),draft_attn_skip_mask)
        model.set_skip_layers(
                attn_skip_layer_id_set=[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],#draft_attn_skip_mask,
                mlp_skip_layer_id_set=[4,5,6,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]#[0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,21,22,23,24,25,26,27,28,30]#,#draft_mlp_skip_mask,
        )
        for _ in range(num_speculations):
            # time1 = time.time()
            with model.self_draft():
                draft_result = model.forward(
                    input_ids=draft_input_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True)
            # time2 = time.time()
            # print(time2-time1)
            # past_key_values = draft_result.past_key_values
            # exit_query_cache = draft_result.exit_query_cache
            draft_logits = draft_result.logits
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)
            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            # print(draft_next_prob.shape,draft_next_token.shape)
            if draft_exiting:# and draft_next_prob.item() < th_stop_draft:
                draft_output_probs = torch.gather(draft_next_prob, -1, draft_next_token).squeeze(-1)
                if draft_output_probs.item() < th_stop_draft and draft_output_ids!=[]:# and (1-random_list[step_draft]) <= th_random_draft) or step + step_draft + 2 >= max_new_tokens:                
                    break                   
            past_key_values = draft_result.past_key_values
            # print("~~~~~~~",type(draft_result.past_key_values),"~~~~~~~~")
            # exit_query_cache = draft_result.exit_query_cache
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            # print(draft_output_ids)
            if sample:
                draft_probabilities.append(draft_next_prob)
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            if draft_next_token == eos_token_id:
                # break out of loop when we get an EOS token
                break

        # input_ids (1 x T_p) and draft_output_ids (1 x T_d) are concatenated together to make
        # 1 x (T_d  + T_p)
        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
        # print(draft_output_ids.shape)
        prefill_token_ids = torch.cat(
            [input_ids, draft_output_ids],
            dim=-1,
        )
        # print(prefill_token_ids.int().shape)
        
        time2 = time.time()
        # print(time2-time1)

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                print(colorama.Fore.LIGHTMAGENTA_EX, end="")
                streamer.put(draft_output_ids, is_draft=True)
                
                
        model.set_skip_layers(
                attn_skip_layer_id_set=None,
                mlp_skip_layer_id_set=None,
        )
        # model.set_skip_layers(
        #         attn_skip_layer_id_set=[6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31],#draft_attn_skip_mask,
        #         mlp_skip_layer_id_set=[0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,21,22,23,24,25,26,27,28,30]#,#draft_mlp_skip_mask,
        # )
        # logits: 1 x (T_d  + T_p) x V
        verify_results = model.forward(
            input_ids=prefill_token_ids.int(),#draft_input_ids,
            past_key_values=past_key_values_predraft,
            return_dict=True,
            use_cache=True)
        logits = verify_results.logits
        # print(logits.shape)
        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        past_key_values = verify_results.past_key_values
        # only select the logits relevant to what the draft has outputted.
        # verification_logits: 1 x T_d x V
        verification_logits = logits[:, prompt_length - 1 :, :]

        # verified_tokens: 1 x (T_d)
        # There is a predicted token for every token in the draft output ids list, however note that the
        # first tokens (or first N tokens) are coming from the prompt
        verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
        # print(len(verified_tokens[0]))
        
        # skip verification of the last token as it is a new token predicted from the main model
        verified_tokens = verified_tokens.to(prefill_token_ids)
        # print(len(draft_output_ids),len(draft_output_ids[0]),len(verified_tokens),len(verified_tokens[0]))
        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]

        # number of matches is the index of the number of tokens we are accepting from the draft
        if not sample:
            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids, dtype=torch.float)
            for i in range(draft_output_ids.numel()):
                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[0, i]].item() / draft_probabilities[i][0, draft_output_ids[0, i]].item()):
                    number_of_matches += 1
                else:
                    verified_tokens[0][number_of_matches] = torch.multinomial(max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()
                    break

        # accept the `number_of_matches` tokens from the draft with one more from the main model
        # since we re-use the same cachem the input id should only be the last accepted token TODO check this
        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]
        output_ids.extend(draft_output_ids[0, : number_of_matches].tolist())
        output_ids.extend(verified_tokens[0][number_of_matches : number_of_matches + 1].tolist())

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                streamer.delete(len(draft_output_ids[0, :]))
                print(colorama.Fore.GREEN, end="")
                streamer.put(draft_output_ids[0, : number_of_matches])
                print(colorama.Style.RESET_ALL, end="")
                streamer.put(verified_tokens[0][number_of_matches : number_of_matches + 1])
            else:
                # streamer.put(torch.cat((draft_output_ids[0, : number_of_matches], verified_tokens[0][number_of_matches : number_of_matches + 1])))
                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))

        # we want the entire output sequence + input sequence
        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids_list) + len(output_ids) - 1
        )
        time3 = time.time()
        # print("-",time3-time2)

        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            draft_output_ids.numel(),
        )
    
    



