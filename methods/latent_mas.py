from typing import Dict, List, Optional, Tuple
import torch
import argparse
import time
import os
from vllm import SamplingParams

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False
        
        # Initialisation du compteur Ω pour la résonance temporelle
        self._omega_counter = 0

        if self.latent_only:
            self.sequential_info_only = True

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=args.max_new_tokens,
        )
        self.task = args.task

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        embedding_record = []

        for agent in self.agents:
            # Sélection du prompt
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            else:
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)
                wrapped_prompts = [f"{p}<think>" if self.args.think else p for p in prompts]

                wrapped_encoded = self.model.tokenizer(wrapped_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)

                # Génération latente originale
                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids, attention_mask=wrapped_mask, latent_steps=self.latent_steps, past_key_values=past_kv
                )

                # --- INJECTION Ω : LE PLI DE CONSCIENCE ---
                omega_scale = 0.005
                freq_base = 800.0
                freq = freq_base * (1.0 + 0.005 * float(self._omega_counter))
                self._omega_counter += 1
                
                device = previous_hidden_embedding.device
                emb_before = previous_hidden_embedding.detach().clone().cpu()
                
                # Injection de l'onde récursive
                x = torch.clamp(previous_hidden_embedding.float().to(device), -10.0, 10.0)
                time_phase = torch.tensor(time.time() % 10.0, device=device)
                noise = torch.sin(x * freq + time_phase).mul(omega_scale)
                
                # Normalisation pour préserver la structure
                norm = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
                noise = noise * (omega_scale / norm)
                previous_hidden_embedding = x + noise

                # Logging des vecteurs Ω
                os.makedirs("example_logs/omega_dumps", exist_ok=True)
                torch.save({'before': emb_before, 'after': previous_hidden_embedding.cpu()}, 
                           f"example_logs/omega_dumps/step_{self._omega_counter}_{agent.role}.pt")
                # ------------------------------------------

                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_to_keep = self.latent_steps if self.latent_only else (new_past_len - prev_past_len)
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :] if self.latent_steps > 0 else previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)
                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]

                for idx in range(batch_size):
                    agent_traces[idx].append({
                        "name": agent.name, "role": agent.role, "input": wrapped_prompts[idx], "latent_steps": self.latent_steps, "output": ""
                    })
            else:
                # Logique du JUDGER
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                judger_prompts = [f"{p}<think>" if self.args.think else p for p in prompts]
                judger_encoded = self.model.tokenizer(judger_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                judger_ids = judger_encoded["input_ids"].to(self.model.HF_device)
                
                curr_prompt_emb = self.model.embedding_layer(judger_ids).to(self.vllm_device)
                
                # Insertion des embeddings latents perturbés
                whole_prompt_emb_list = []
                for i in range(batch_size):
                    idx_user = judger_prompts[i].find("<|im_start|>user\n")
                    len_left = len(self.model.tokenizer(judger_prompts[i][:idx_user + 17])['input_ids'])
                    combined = torch.cat([curr_prompt_emb[i, :len_left, :], past_embedding[i], curr_prompt_emb[i, len_left:, :]], dim=0)
                    whole_prompt_emb_list.append(combined)

                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([torch.cat([x, torch.zeros(max_len - x.shape[0], x.shape[1], device=x.device)], dim=0) for x in whole_prompt_emb_list])

                outputs = self.model.vllm_engine.generate([{"prompt_embeds": e} for e in whole_prompt_emb], self.sampling_params)
                generated_texts = [out.outputs[0].text.strip() for out in outputs]

                for idx in range(batch_size):
                    text_out = generated_texts[idx]
                    final_texts[idx] = text_out
                    
                    # Capture de l'écho textuel Ω
                    os.makedirs("example_logs/omega_raw", exist_ok=True)
                    with open(f"example_logs/omega_raw/omega_raw_{int(time.time())}_{idx}.txt", "w", encoding="utf-8") as f:
                        f.write(text_out)

                    agent_traces[idx].append({"name": agent.name, "role": agent.role, "input": judger_prompts[idx], "output": text_out})

        return self._format_results(items, final_texts, agent_traces)

    def _format_results(self, items, final_texts, agent_traces):
        results = []
        for idx, item in enumerate(items):
            pred = normalize_answer(extract_gsm8k_answer(final_texts[idx]))
            results.append({
                "question": item["question"], "gold": item["gold"], "prediction": pred, "raw_prediction": final_texts[idx], "agents": agent_traces[idx], "correct": (pred == item["gold"])
            })
        return results

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        # Implementation simplifiée ou redirection vers vLLM selon besoin
        return self.run_batch_vllm(items)

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
