import os
import random
import time
import math
import functools
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional, Tuple, Union
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
import deepspeed
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedModel

from openrlhf.models import Actor
from openrlhf.utils import get_strategy

torch.set_printoptions(threshold=10_000)


@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams:
    use_adaptive_kl: bool = False
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)
    kl_coef: float = 0.05


@dataclass
class REBELHParams:
    num_updates: int = 1000
    noptepochs: int = 4
    whiten_rewards: bool = False
    shift_mean: bool = False
    eta: float = 1.0


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 1024
    query_dataset: str = "GitBag/ultrafeedback_llama3_eurus"

    # Response params
    response_length: int = 1024
    penalty_reward_value: int = -4
    reward_reg: float = 1000

    # LM params
    temperature: float = 0.5


@dataclass
class Args:
    # common args
    exp_name: str = "ultrafeedback_rebel"
    seed: int = 555134
    track: bool = True
    wandb_project_name: str = "ultrafeedback"
    cuda: bool = True
    run_name: Optional[str] = None
    push_to_hub: bool = False
    hf_entity: str = ""
    deepspeed: bool = True
    print_sample_output_freq: int = 200
    run_eval: bool = True

    # optimizer args
    eps: float = 1e-8
    lr: float = 1e-7
    weight_decay: float = 1e-6
    optimizer: Literal["adam", "adamw"] = "adamw"
    scheduler: str = "linear"
    warm_up_steps: int = 0

    gradient_accumulation_steps: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 16
    per_device_reward_batch_size: int = 2
    total_episodes: int = 1_000_000

    # optional args filled while running
    world_size: Optional[int] = 4
    batch_size: Optional[int] = 512
    local_rollout_forward_batch_size: int = 16
    local_batch_size: Optional[int] = 128

    # other args
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    offload: bool = True
    reward_model: str = "openbmb/Eurus-RM-7b"
    dropout_layer_keys: List[str] = field(
        default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"]
    )
    output_dir: str = "models/rebel_ultrafeedback"
    num_layers_unfrozen: int = 4
    task: TaskHParams = field(default_factory=TaskHParams)
    reward: RewardHParams = field(default_factory=RewardHParams)
    rebel: REBELHParams = field(default_factory=REBELHParams)

    # Deepspeed / strategy args for OpenRLHF integration
    max_norm: float = 1.0
    full_determinism: bool = False
    micro_train_batch_size: int = 2
    train_batch_size: int = 128
    zero_stage: int = 2
    bf16: bool = True
    local_rank: int = -1
    zpg: int = 1
    adam_offload: bool = False
    use_ds_universal_ckpt: bool = False
    grad_accum_dtype: Optional[str] = None
    overlap_comm: bool = False
    deepcompile: bool = False
    ds_tensor_parallel_size: int = 1
    ring_attn_size: int = 1


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)
    return table


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


def whiten(values, shift_mean=True):
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args: Args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == tokenizer.eos_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [args.task.response_length]
    idxs = torch.arange(args.task.response_length, device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def rhasattr(obj, attr):
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args) -> object:
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")


def hf_get_decoder_final_norm(model: nn.Module) -> float:
    norm_attrs = (
        "transformer.ln_f",
        "model.decoder.final_layer_norm",
        "model.norm",
        "decoder.final_layer_norm",
        "gpt_neox.final_layer_norm",
    )
    return findattr(model, norm_attrs)


def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    hidden_layers_attrs = (
        "h",
        "layers",
        "model.layers",
        "decoder.layers",
        "transformer.h",
        "transformer.blocks",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
    )
    return findattr(model, hidden_layers_attrs)


def hf_get_lm_head(model: nn.Module) -> nn.Module:
    return model.get_output_embeddings()


def hf_get_hidden_size(config: PretrainedConfig) -> int:
    hidden_size_attrs = ("hidden_size", "n_embd", "d_model")
    return findattr(config, hidden_size_attrs)


class ModelBranch(PreTrainedModel):
    def __init__(
        self,
        base_model: PreTrainedModel,
        *,
        num_layers_unfrozen: int,
        frozen: bool = True,
    ):
        config = base_model.config
        super().__init__(config)

        decoder_blocks = hf_get_decoder_blocks(base_model)[-num_layers_unfrozen:]
        final_norm = hf_get_decoder_final_norm(base_model)
        lm_head = hf_get_lm_head(base_model)

        with deepspeed.zero.GatheredParameters(
            list(decoder_blocks.parameters()) + list(final_norm.parameters()) + list(lm_head.parameters()),
            modifier_rank=None,
        ):
            self.decoder_blocks = deepcopy(decoder_blocks)
            self.final_norm = deepcopy(final_norm)
            self.lm_head = deepcopy(lm_head)

        self.hidden_size = hf_get_hidden_size(self.config)
        self.model_parallel = False
        self.device_map = None
        self.last_device = None
        self.gradient_checkpointing = False

        if frozen:
            for parameter in self.parameters():
                parameter.requires_grad_(False)


class LlamaBranch(ModelBranch):
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
    ):
        past_seen_tokens = 0
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_shape: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        _, seq_length = hidden_states.shape[:2]

        past_seen_tokens = 0

        device = hidden_states.device
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position)

        for decoder_layer in self.decoder_blocks:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return logits


def freeze_bottom_causal_layers(model, num_layers_unfrozen: int = 0):
    def hf_get_decoder_blocks(model: nn.Module):
        hidden_layers_attrs = (
            "h",
            "layers",
            "model.layers",
            "decoder.layers",
            "transformer.h",
            "transformer.blocks",
            "model.decoder.layers",
            "gpt_neox.layers",
            "decoder.block",
        )
        return findattr(model, hidden_layers_attrs)

    hidden_layers = hf_get_decoder_blocks(model)

    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
        hidden_layers_to_freeze += [model.get_input_embeddings(), model.get_output_embeddings()]
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
        hidden_layers_to_freeze += [model.get_input_embeddings()]
        if model.config.tie_word_embeddings:
            hidden_layers_to_freeze += [model.get_output_embeddings()]
    else:
        hidden_layers_to_freeze = []

    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


@torch.no_grad()
def get_reward(
    reward_model,
    query_tokens,
    response_tokens,
    tokenizer,
    rm_tokenizer,
    device,
    reward_batch_size,
):
    prompt = rm_tokenizer.batch_decode(query_tokens, skip_special_tokens=False)
    responses = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
    text = [" ".join([p, r]) for p, r in zip(prompt, responses)]

    old_side = rm_tokenizer.truncation_side
    rm_tokenizer.truncation_side = "left"
    rm_response_tokens = rm_tokenizer(
        text,
        padding="max_length",
        max_length=2048,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    rm_tokenizer.truncation_side = old_side

    input_ids = rm_response_tokens["input_ids"].to(device)
    attention_mask = input_ids != rm_tokenizer.pad_token_id
    input_ids = torch.masked_fill(input_ids, ~attention_mask, rm_tokenizer.eos_token_id)
    out = []
    mbs = reward_batch_size
    for i in range(math.ceil(input_ids.shape[0] / mbs)):
        rewards = reward_model(
            input_ids=input_ids[i * mbs : (i + 1) * mbs],
            attention_mask=attention_mask[i * mbs : (i + 1) * mbs],
        )
        out.extend(rewards)
    return torch.hstack(out)


@torch.no_grad()
def generate(lm_backbone, queries, tokenizer, generation_config):
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, tokenizer.eos_token_id)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


@dataclass
class EvalStorage:
    query: List[str] = field(default_factory=list)
    postprocessed_response: List[str] = field(default_factory=list)
    score: List[float] = field(default_factory=list)


def evaluate(args: Args, reward_model, policy, tokenizer, rm_tokenizer, dataloader, generation_config, sampling=True):
    eval_storage = EvalStorage()
    device = torch.cuda.current_device()
    with torch.no_grad():
        for data in tqdm(dataloader):
            query = data["llama_prompt_tokens"].to(device)
            rm_query = data["eurus_prompt_tokens"]
            context_length = query.shape[1]
            query_responses = generate(
                policy,
                query,
                tokenizer,
                generation_config,
            )
            responses = query_responses[:, context_length:]
            postprocessed_responses = truncate_response(args, tokenizer, responses)
            score = get_reward(
                reward_model,
                rm_query,
                postprocessed_responses,
                tokenizer,
                rm_tokenizer,
                device,
                args.per_device_reward_batch_size,
            )
            torch.cuda.empty_cache()

            eval_storage.query.extend(data["instruction"])
            eval_storage.postprocessed_response.extend(
                tokenizer.batch_decode(postprocessed_responses, skip_special_tokens=False)
            )
            eval_storage.score.append(score)

            if sampling:
                break

    eval_score = torch.cat(eval_storage.score).float().cpu().numpy().tolist()
    eval_df = pd.DataFrame(
        {
            "query": eval_storage.query,
            "postprocessed_response": eval_storage.postprocessed_response,
            "scores": eval_score,
        }
    )
    return eval_storage, eval_df


def train_rebel(
    args: Args,
    strategy=None,
    actor: Optional[Actor] = None,
    tokenizer=None,
    rm_tokenizer=None,
    reward_model=None,
    dataloader=None,
    validation_dataloader=None,
) -> None:
    if strategy is None:
        strategy = get_strategy(args)
        strategy.setup_distributed()

    local_seed = args.seed + strategy.get_rank() * 100003

    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    args.world_size = strategy.world_size
    args.batch_size = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.rebel.whiten_rewards:
        assert (
            args.local_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_batch_size} is insufficient for whitening"
    args.rebel.num_updates = args.total_episodes // args.batch_size

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}__{args.output_dir.split('/')[1]}"
    print("Wandb run name: ", run_name)
    writer = SimpleNamespace()
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z, max_bins: None
    if strategy.is_rank_0():
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = torch.cuda.current_device()
    torch.backends.cudnn.deterministic = True

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="right", trust_remote_code=True)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if rm_tokenizer is None:
        rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model, padding_side="right", trust_remote_code=True)
        rm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if actor is None:
        # Use OpenRLHF Actor wrapper so that future steps can reuse the same
        # acceleration features (LoRA, quantization, etc.) as other trainers.
        actor = Actor(
            args.base_model,
            attn_implementation="eager",
            bf16=args.bf16,
            load_in_4bit=False,
            lora_rank=0,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=None,
            ds_config=None,
            packing_samples=False,
            use_liger_kernel=False,
        )
    policy = actor.model
    ref_policy = LlamaBranch(policy, num_layers_unfrozen=args.num_layers_unfrozen)

    freeze_bottom_causal_layers(policy, num_layers_unfrozen=args.num_layers_unfrozen)
    policy.generation_config.eos_token_id = None
    policy.generation_config.pad_token_id = None
    policy.generation_config.do_sample = True

    if reward_model is None:
        reward_model = AutoModel.from_pretrained(args.reward_model, trust_remote_code=True)
        reward_model.eval().requires_grad_(False)

    if dataloader is None or validation_dataloader is None:
        dataset = load_dataset(args.task.query_dataset, split="train")
        dataset = dataset.with_format("torch", columns=["llama_prompt_tokens", "eurus_prompt_tokens"])
        dataloader = strategy.setup_dataloader(
            dataset,
            args.local_batch_size,
            pin_memory=False,
            shuffle=True,
        )
        validation_dataset = load_dataset(args.task.query_dataset, split="train")
        validation_dataset = validation_dataset.with_format(
            "torch", columns=["llama_prompt_tokens", "eurus_prompt_tokens", "instruction"]
        )
        validation_dataloader = strategy.setup_dataloader(
            validation_dataset,
            args.per_device_eval_batch_size,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

    if strategy.is_rank_0():
        pprint(policy.config)
        pprint(reward_model.config)

    optimizer = strategy.create_optimizer(actor, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    kl_ctl = AdaptiveKLController(args.reward.kl_coef, hparams=args.reward.adaptive_kl)
    generation_config = GenerationConfig(
        min_new_tokens=args.task.response_length,
        max_new_tokens=args.task.response_length,
        temperature=(args.task.temperature + 1e-7),
        top_p=1.0,
        top_k=0,
        do_sample=True,
    )
    validation_generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(0.01 + 1e-7),
        do_sample=True,
    )

    torch.manual_seed(args.seed)
    (actor, optimizer, _) = strategy.prepare((actor, optimizer, None), is_rlhf=True)
    policy = actor.model

    def repeat_generator():
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    torch.manual_seed(local_seed)

    if args.deepspeed:
        # Wrap reward and reference models with Deepspeed eval engines via strategy
        reward_model._offload = args.offload  # type: ignore[attr-defined]
        ref_policy._offload = args.offload  # type: ignore[attr-defined]
        reward_model = strategy._ds_init_eval_model(reward_model)
        reward_model.eval()
        ref_policy = strategy._ds_init_eval_model(ref_policy)
        ref_policy.eval()
    else:
        reward_model = reward_model.to(device)
        ref_policy = ref_policy.to(device)

    strategy.print("===training policy===")
    global_step = 0
    start_time = time.time()
    stats_shape = (args.rebel.noptepochs, args.gradient_accumulation_steps)

    approxkl_stats = torch.zeros(stats_shape, device=device)
    loss_stats = torch.zeros((args.rebel.noptepochs, args.gradient_accumulation_steps), device=device)
    entropy_stats = torch.zeros(stats_shape, device=device)
    ratio_stats = torch.zeros(stats_shape, device=device)

    policy.train()
    for update in range(1, args.rebel.num_updates + 1):
        global_step += 1 * args.batch_size
        frac = 1.0 - (update - 1.0) / args.rebel.num_updates
        lrnow = frac * args.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        with torch.no_grad():
            eval_storage, eval_df = evaluate(
                args,
                reward_model,
                strategy._unwrap_model(actor),
                tokenizer,
                rm_tokenizer,
                validation_dataloader,
                validation_generation_config,
            )
            validation_score = eval_storage.score[0]
            if args.print_sample_output_freq > 0 and update > 1 and (update - 1) % args.print_sample_output_freq == 0:
                if strategy.is_rank_0():
                    eval_df.to_csv(f"runs/{run_name}/table_{global_step}.csv")
                    if args.track:
                        wandb.log({"samples/query_responses": wandb.Table(dataframe=eval_df)}, step=update)
                    else:
                        try:
                            print_rich_table(f"Sample Output at Step {update}", eval_df[:1], console)
                        except Exception as e:
                            print(e)
                if args.output_dir:
                    output_dir = os.path.join(args.output_dir, run_name, str(update))
                    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                    time_tensor = torch.tensor([int(time.time())], device=device)
                    time_int = time_tensor[0].item()
                    repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
                    repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

                    if strategy.is_rank_0():
                        tokenizer.save_pretrained(output_dir)
                        if args.push_to_hub:
                            tokenizer.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}")

                    unwrapped: PreTrainedModel = strategy._unwrap_model(actor)  # type: ignore[assignment]
                    if strategy.is_rank_0():
                        unwrapped.save_pretrained(
                            output_dir,
                            is_main_process=True,
                            save_function=torch.save,
                            state_dict=unwrapped.state_dict(),
                            safe_serialization=False,
                            repo_id=repo_id,
                        )
                        if args.push_to_hub:
                            unwrapped.push_to_hub(
                                repo_id, revision=f"seed{args.seed}_{str(time_int)}", safe_serialization=False
                            )
            del eval_storage, eval_df
            torch.cuda.empty_cache()

            queries = data["llama_prompt_tokens"].to(device)
            rm_queries = data["eurus_prompt_tokens"]
            context_length = queries.shape[1]

            query_responses = []
            responses = []
            postprocessed_responses = []
            logprobs = []
            ref_logprobs = []
            scores = []
            sequence_lengths = []

            for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                query = queries[i : i + args.local_rollout_forward_batch_size]
                rm_query = rm_queries[i : i + args.local_rollout_forward_batch_size]

                batch_query_responses = []
                batch_responses = []
                batch_postprocessed_responses = []
                batch_logprobs = []
                batch_ref_logprobs = []
                batch_scores = []
                batch_sequence_lengths = []

                for _ in range(2):
                    query_response = generate(
                        strategy._unwrap_model(actor),
                        query,
                        tokenizer,
                        generation_config,
                    )
                    response = query_response[:, context_length:]

                    attention_mask = query_response != tokenizer.pad_token_id
                    input_ids = torch.masked_fill(query_response, ~attention_mask, tokenizer.eos_token_id)
                    output = policy(
                                 input_ids=input_ids, 
                                 attention_mask=attention_mask,
                        return_dict=True,
                        output_hidden_states=True,
                    )

                    input_hidden_states = output.hidden_states[-(args.num_layers_unfrozen + 1)]
                    output_shape = output.hidden_states[-1].size()

                    logits = output.logits[:, context_length - 1 : -1]
                    logits /= args.task.temperature + 1e-7
                    all_logprob = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del output, logits, all_logprob, input_ids
                    torch.cuda.empty_cache()

                    ref_logits = ref_policy(
                        hidden_states=input_hidden_states,
                        output_shape=output_shape,
                        attention_mask=attention_mask,
                    )
                    ref_logits = ref_logits[:, context_length - 1 : -1]
                    ref_logits /= args.task.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del ref_logits, ref_all_logprob, input_hidden_states, output_shape, attention_mask
                    torch.cuda.empty_cache()

                    postprocessed_response = truncate_response(args, tokenizer, response)

                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    score = get_reward(
                        reward_model,
                        rm_query,
                        postprocessed_response,
                        tokenizer,
                        rm_tokenizer,
                        device,
                        args.per_device_reward_batch_size,
                    )

                    batch_query_responses.append(query_response)
                    batch_responses.append(response)
                    batch_postprocessed_responses.append(postprocessed_response)
                    batch_logprobs.append(logprob)
                    batch_ref_logprobs.append(ref_logprob)
                    batch_scores.append(score)
                    batch_sequence_lengths.append(sequence_length)

                query_responses.append(torch.stack(batch_query_responses, 1))
                responses.append(torch.stack(batch_responses, 1))
                postprocessed_responses.append(torch.stack(batch_postprocessed_responses, 1))
                logprobs.append(torch.stack(batch_logprobs, 1))
                ref_logprobs.append(torch.stack(batch_ref_logprobs, 1))
                scores.append(torch.stack(batch_scores, 1))
                sequence_lengths.append(torch.stack(batch_sequence_lengths, 1))

            query_responses = torch.cat(query_responses, 0).flatten(end_dim=1)
            responses = torch.cat(responses, 0).flatten(end_dim=1)
            postprocessed_responses = torch.cat(postprocessed_responses, 0).flatten(end_dim=1)
            logprobs = torch.cat(logprobs, 0).flatten(end_dim=1)
            ref_logprobs = torch.cat(ref_logprobs, 0).flatten(end_dim=1)
            scores = torch.cat(scores, 0).flatten(end_dim=1)
            sequence_lengths = torch.cat(sequence_lengths, 0).flatten(end_dim=1)
            del (
                logprob,
                ref_logprob,
                score,
                batch_query_responses,
                batch_responses,
                batch_postprocessed_responses,
                batch_logprobs,
                batch_ref_logprobs,
                batch_scores,
                batch_sequence_lengths,
            )
            torch.cuda.empty_cache()

            scores = scores / args.task.reward_reg

            contain_pad_token = torch.any(postprocessed_responses == tokenizer.pad_token_id, dim=-1)
            scores = torch.where(contain_pad_token, scores, torch.full_like(scores, args.task.penalty_reward_value))
            strategy.print(f"{scores=}, {(contain_pad_token.sum() / len(contain_pad_token))=}")

            seq_mask = (
                torch.arange(responses.size(1), device=policy.device)
                .unsqueeze(0)
                .expand_as(responses)
                <= sequence_lengths.unsqueeze(1)
            )
            logprobs = (logprobs * seq_mask).sum(-1)
            ref_logprobs = (ref_logprobs * seq_mask).sum(-1)

            kl = logprobs - ref_logprobs
            non_score_reward = -kl_ctl.value * kl
            rewards = non_score_reward + scores
            if args.rebel.whiten_rewards:
                rewards = whiten(rewards, args.rebel.shift_mean)

            strategy.print("rewards with kl====", rewards)
            if strategy.is_rank_0():
                console.print(
                    "mean_kl",
                    kl.mean().item(),
                    "scores",
                    scores.mean().item(),
                )
            del sequence_lengths, ref_logprobs, postprocessed_responses
            torch.cuda.empty_cache()

        for rebel_epoch_idx in range(args.rebel.noptepochs):
            local_batch_idxs = np.random.permutation(args.local_batch_size)
            gradient_accumulation_idx = 0
            for mini_batch_start in range(0, args.local_batch_size, args.per_device_train_batch_size):
                mini_batch_end = mini_batch_start + args.per_device_train_batch_size
                mini_batch_inds = local_batch_idxs[mini_batch_start:mini_batch_end] * 2
                mini_batch_inds = np.append(mini_batch_inds, mini_batch_inds + 1)
                mb_responses = responses[mini_batch_inds]
                mb_query_responses = query_responses[mini_batch_inds]
                mb_logprobs = logprobs[mini_batch_inds]
                mb_rewards = rewards[mini_batch_inds]
                mb_seq_mask = seq_mask[mini_batch_inds]

                attention_mask = mb_query_responses != tokenizer.pad_token_id
                mb_input_ids = torch.masked_fill(mb_query_responses, ~attention_mask, tokenizer.eos_token_id)
                output = policy(
                    input_ids=mb_input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )
                logits = output.logits[:, context_length - 1 : -1]
                logits /= args.task.temperature + 1e-7
                new_all_logprobs = F.log_softmax(logits, dim=-1)
                new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                new_logprobs = (new_logprobs * mb_seq_mask).sum(-1)

                ratio_logprob = new_logprobs - mb_logprobs
                ratio_logprob = ratio_logprob[: args.per_device_train_batch_size] - ratio_logprob[
                    args.per_device_train_batch_size :
                ]
                reg_diff = ratio_logprob - args.rebel.eta * (
                    mb_rewards[: args.per_device_train_batch_size] - mb_rewards[args.per_device_train_batch_size :]
                )
                loss = (reg_diff**2).mean()

                strategy.backward(loss, actor, optimizer)
                strategy.optimizer_step(optimizer, actor, None)
                optimizer.zero_grad()
                with torch.no_grad():
                    y = args.rebel.eta * (
                        mb_rewards[: args.per_device_train_batch_size] - mb_rewards[args.per_device_train_batch_size :]
                    )
                    logprobs_diff = new_logprobs - mb_logprobs
                    ratio = torch.exp(logprobs_diff)
                    prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                    approxkl = 0.5 * (logprobs_diff**2).mean()
                    approxkl_stats[rebel_epoch_idx, gradient_accumulation_idx] = approxkl
                    loss_stats[rebel_epoch_idx, gradient_accumulation_idx] = loss
                    entropy_stats[rebel_epoch_idx, gradient_accumulation_idx] = entropy.mean()
                    ratio_stats[rebel_epoch_idx, gradient_accumulation_idx] = ratio.mean()
                gradient_accumulation_idx += 1
            if strategy.is_rank_0():
                console.print(
                    "rebel_epoch_idx",
                    rebel_epoch_idx,
                    "approxkl",
                    approxkl_stats[rebel_epoch_idx].mean().item(),
                    "loss",
                    loss_stats[rebel_epoch_idx].mean().item(),
                )

        with torch.no_grad():
            mean_kl = kl.mean()
            mean_entropy = -logprobs.mean()
            mean_non_score_reward = non_score_reward.mean()
            writer.add_scalar("objective/kl_coef", kl_ctl.value, update)
            writer.add_scalar("objective/kl", mean_kl.item(), update)
            writer.add_scalar("objective/entropy", mean_entropy.item(), update)
            writer.add_scalar("objective/non_score_reward", mean_non_score_reward.item(), update)
            writer.add_scalar(
                "objective/score_total",
                (mean_non_score_reward + scores.mean()).item(),
                update,
            )
            writer.add_scalar("objective/scores", scores.mean().item(), update)
            writer.add_scalar("objective/validation_score", validation_score.mean().item(), update)
            writer.add_histogram(
                "objective/scores_his",
                scores.cpu().float().numpy().flatten(),
                update,
                max_bins=64,
            )
            writer.add_histogram(
                "objective/validation_scores_his",
                validation_score.cpu().float().numpy().flatten(),
                update,
                max_bins=64,
            )
            writer.add_scalar("npg/loss/policy", loss.item(), update)
            writer.add_scalar("npg/policy/entropy", entropy.mean().item(), update)
            writer.add_scalar("npg/policy/approxkl", approxkl.item(), update)

            writer.add_scalar("npg/policy/initial_loss", loss_stats[0].mean().item(), update)
            writer.add_scalar("npg/policy/final_loss", loss_stats[-1].mean().item(), update)
            writer.add_scalar(
                "npg/policy/delta_loss",
                (loss_stats[-1] - loss_stats[0]).mean().item(),
                update,
            )

            writer.add_scalar("npg/policy/approxkl_avg", approxkl_stats.mean().item(), update)
            writer.add_scalar("npg/loss/policy_avg", loss_stats.mean().item(), update)
            writer.add_scalar("npg/policy/entropy_avg", entropy_stats.mean().item(), update)
            writer.add_scalar("npg/val/ratio", ratio_stats.mean().item(), update)
            writer.add_scalar("npg/val/ratio_var", ratio_stats.var().item(), update)
            writer.add_scalar("npg/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)
            writer.add_scalar("npg/lr", lrnow, update)
            writer.add_scalar("npg/episode", global_step, update)
            eps = int(global_step / (time.time() - start_time))
            writer.add_scalar("npg/eps", eps, update)
            strategy.print("npg/eps", eps, update)
            if args.reward.use_adaptive_kl:
                kl_ctl.update(mean_kl.item(), args.batch_size)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores
            torch.cuda.empty_cache()

    if args.run_eval:
        eval_storage, eval_df = evaluate(
            args,
            reward_model,
            strategy._unwrap_model(actor),
            tokenizer,
            rm_tokenizer,
            validation_dataloader,
            validation_generation_config,
        )
        if strategy.is_rank_0():
            eval_df.to_csv(f"runs/{run_name}/table.csv")
            if args.track:
                wandb.log({"samples/query_responses": wandb.Table(dataframe=eval_df)}, step=update)

    if args.output_dir:
        output_dir = os.path.join(args.output_dir, run_name, str(update))
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        time_tensor = torch.tensor([int(time.time())], device=device)
        time_int = time_tensor[0].item()
        repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

        if strategy.is_rank_0():
            tokenizer.save_pretrained(output_dir, repo_id=repo_id)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}")

        unwrapped: PreTrainedModel = strategy._unwrap_model(actor)  # type: ignore[assignment]
        if strategy.is_rank_0():
            unwrapped.save_pretrained(
                output_dir,
                is_main_process=True,
                save_function=torch.save,
                state_dict=unwrapped.state_dict(),
                safe_serialization=False,
                repo_id=repo_id,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(
                    repo_id,
                    revision=f"seed{args.seed}_{str(time_int)}",
                    safe_serialization=False,
                )


class REBELTrainer:
    def __init__(
        self,
        args: Optional[Args] = None,
        strategy=None,
        actor: Optional[Actor] = None,
        tokenizer=None,
        rm_tokenizer=None,
        reward_model=None,
        train_dataloader=None,
        validation_dataloader=None,
    ) -> None:
        self.args = args or Args()
        self.strategy = strategy
        self.actor = actor
        self.tokenizer = tokenizer
        self.rm_tokenizer = rm_tokenizer
        self.reward_model = reward_model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def fit(self) -> None:
        train_rebel(
            self.args,
            strategy=self.strategy,
            actor=self.actor,
            tokenizer=self.tokenizer,
            rm_tokenizer=self.rm_tokenizer,
            reward_model=self.reward_model,
            dataloader=self.train_dataloader,
            validation_dataloader=self.validation_dataloader,
        )


def main() -> None:
    args = tyro.cli(Args)
    train_rebel(args)


if __name__ == "__main__":
    main()
