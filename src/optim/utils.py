from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import torch.distributed as dist
import math
import wandb


def get_batch(datareader, device="cpu"):
    x, y = datareader.sample_batch()
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def cos_inf_schedule(n_iterations, n_warmup, div_factor, final_div_factor, n_inf):
    """Cosine annealing with warmup and _constant_ final_lr after cycle ended.
    Args:
        n_iterations: total number of iterations
        n_warmup: number of warmup iterations
        div_factor: initial division factor for warmup
        final_div_factor: final division factor for final lr
        n_inf: number of iterations for the final lr (constant lr after cycle ended)
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    max_lr = 1.0
    base_lr = max_lr / div_factor
    final_lr = base_lr / final_div_factor

    n_anneal_steps = n_iterations - n_inf

    def schedule(step):
        # Warmup phase: Step < n_warmup
        if step < n_warmup:
            # Linearly increase LR from base_lr to max_lr
            return (step / n_warmup) + (1 - step / n_warmup) / div_factor
        # Cosine annealing phase: n_warmup ≤ Step < n_anneal_steps
        elif step < n_anneal_steps:
            # Compute progress in cosine decay range (normalized 0 to 1)
            t = (step - n_warmup) / (n_anneal_steps - n_warmup)
            # Compute cosine decay learning rate
            lr = final_lr + 0.5 * (max_lr - final_lr) * (1 + np.cos(np.pi * t))
            return lr
        # Final constant phase: Step ≥ n_anneal_steps
        else:
            return final_lr

    return schedule


def wsd_schedule(
    n_iterations,
    final_lr_factor=0.0,
    n_warmup=1000,
    init_div_factor=100,
    fract_decay=0.1,
    decay_type="linear",
):
    """Warmup, hold, and decay schedule.
    Args:
        n_iterations: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        warmup_fract: fraction of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    n_anneal_steps = int(fract_decay * n_iterations) # Number of decay steps
    n_hold = n_iterations - n_anneal_steps # Hold phase (constant LR)

    def schedule(step):
        if step < n_warmup: # (1) Warmup Phase
            return (step / n_warmup) + (1 - step / n_warmup) / init_div_factor
        elif step < n_hold: # (2) Hold Phase
            return 1.0
        elif step < n_iterations: # (3) Decay Phase
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
            elif decay_type == "exp":
                return final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                return (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "miror_cosine":
                cosine_value = (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
                linear_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - ((step - n_hold) / n_anneal_steps) ** 2
                )

            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - math.sqrt((step - n_hold) / n_anneal_steps)
                )

            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )

        else:
            return final_lr_factor

    return schedule


@torch.no_grad()
def eval(
    model,
    reader,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
    moe=False,
    get_router_logits=False,
    cfg=None,
):
    assert model.training == False

    loss_list_val, acc_list, loss_list_aux_val = [], [], {}

    router_logits = []
    selected_experts_global_list = []
    for idx in range(max_num_batches):
        x, y = get_batch(reader, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True, moe=moe)
        val_loss = outputs["loss"]

        loss_list_val.append(val_loss)
        acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())   #get highest logit and compare with the target 

        # auxiliary losses are optional
        for k, v in outputs["aux_losses"].items():
            loss_list_aux_val[k] = loss_list_aux_val.get(k, [])
            loss_list_aux_val[k].append(v)

        # router logits for MoE visualization
        if get_router_logits:
            # shape [layers, batch_size * sequence_length, num_experts]
            logits = outputs["router_logits"]
            # shape [max_batches, layers, batch_size * sequence_length, num_experts]
            router_logits.append(logits)

        if moe and "selected_experts" in outputs:
            # outputs["selected_experts"] is a list (one per MoE layer)
            if not selected_experts_global_list:
                # Create a list for each layer.
                selected_experts_global_list = [[] for _ in outputs["selected_experts"]]
            for i, se in enumerate(outputs["selected_experts"]):
                selected_experts_global_list[i].append(se)

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828**val_loss
    val_aux_losses = {
        f"val/{k}": torch.stack(v).mean().item() for k, v in loss_list_aux_val.items()
    }

    # Compute global maxvio if we collected selections.
    avg_maxvio_global = None
    per_layer_maxvio_global = {}
    if moe and selected_experts_global_list:
        num_non_shared_experts = cfg.moe_num_experts - cfg.moe_num_shared_experts if cfg else 1
        maxvio_global_list = []
        for layer_idx, se_list in enumerate(selected_experts_global_list):
            concat_tensor = torch.cat(se_list, dim=0)  # Concatenate along token dimension.
            maxvio, load = compute_maxvio(concat_tensor, num_non_shared_experts)
            maxvio_global_list.append(maxvio)
            per_layer_maxvio_global[f"val/maxvioglobal_layer_{layer_idx}"] = maxvio.item()
        avg_maxvio_global = sum(maxvio_global_list) / len(maxvio_global_list)


    if get_router_logits:
        # filter out the router logits that are not of the expected shape (happens for the last batch in
        # dataloader has a different batch size than the others)
        if cfg:
            intended_size = cfg.batch_size * cfg.sequence_length
        else:
            intended_size = x.shape[0] * x.shape[1]
        # shape [batches - 1, layers, batch_size * sequence_length, num_experts]
        router_logits = (
            torch.stack(
                [rl for rl in router_logits if rl.shape[1] == intended_size],
                dim=0,
            )
            .detach()
            .cpu()
        )

    return val_acc, val_loss, val_perplexity, val_aux_losses, router_logits, avg_maxvio_global, per_layer_maxvio_global


@torch.no_grad()
def eval_sweep_dropk(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    n_heads,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    x_axis, y_axis_pp, y_axis_acc, y_axis_loss = (
        torch.linspace(0.0, 0.95, 15),
        [],
        [],
        [],
    )
    loss_list_val, acc_list = [], []

    for frac in x_axis:
        drop_k = int(sequence_length * frac * n_heads)
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=None, drop_k=drop_k, get_logits=True
                )
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


@torch.no_grad()
def eval_sweep_alphath(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    alpha_ths, y_axis_pp, y_axis_acc, y_axis_loss = (
        [0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1],
        [],
        [],
        [],
    )
    loss_list_val, acc_list, x_axis = [], [], []

    for alpha_th in alpha_ths:
        frac_heads_pruned_list = []
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=alpha_th, drop_k=None, get_logits=True
                )
            nph, nh = (
                outputs["num_head_pruned_per_layer"],
                outputs["num_heads_per_layer"],
            )
            frac_heads_pruned = np.sum(nph) / np.sum(
                nh
            )  # fractions of heads removed given alpha_th
            frac_heads_pruned_list.append(frac_heads_pruned)
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        x_axis.append(np.mean(frac_heads_pruned_list))
        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


def save_checkpoint(model, opt, scheduler, itr, ckpt_dir: Path, wandb_run: None):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    # Use the correct optimizer state dict method
    if hasattr(opt, "distributed_state_dict"):
        optimizer_state = opt.distributed_state_dict(
            key_to_param=model.named_parameters()
        )
    else:
        optimizer_state = opt.state_dict()

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer_state,
        "scheduler": scheduler.state_dict(),
        "itr": itr,
    }
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint, ckpt_dir / "main.pt")

    if wandb_run is not None and wandb_run:
        # Save the wandb run ID so that we can resume the same run if needed
        wandb_run_id_file = ckpt_dir / "wandb_id.txt"
        wandb_run_id_file.parent.mkdir(parents=True, exist_ok=True)
        with open(wandb_run_id_file, "w") as f:
            f.write(wandb.run.id)


def load_checkpoint(model, opt, scheduler, ckpt_path, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"])
    scheduler.load_state_dict(ckpt["scheduler"])
    itr = ckpt["itr"]

    # Use the special loader if it's DistributedShampoo
    if hasattr(opt, "load_distributed_state_dict"):
        opt.load_distributed_state_dict(
            state_dict=ckpt["optimizer"],
            key_to_param=model.named_parameters()
        )
    else:
        opt.load_state_dict(ckpt["optimizer"])

    return itr

def save_worker_state(ckpt_dir: Path):
    # Dataloader, rng states
    worker_state = {
        "rng_torch_cpu": torch.random.get_rng_state(),
        "rng_torch_gpu": torch.cuda.get_rng_state(),
        "rng_np": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(worker_state, ckpt_dir / f"worker_{rank}.pt")


def load_worker_state(ckpt_dir: Path):
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    worker_state = torch.load(ckpt_dir / f"worker_{rank}.pt")
    torch.random.set_rng_state(worker_state["rng_torch_cpu"])
    torch.cuda.set_rng_state(worker_state["rng_torch_gpu"])
    np.random.set_state(worker_state["rng_np"])
    random.setstate(worker_state["rng_python"])

def visualize_routing(router_logits, extra_args):
    # router_logits: [batches, layers, batch_size * sequence_length, num_experts]
    logs = {}

    n_layers = extra_args.n_layer
    num_experts = extra_args.moe_num_experts
    num_experts_per_tok = extra_args.moe_num_experts_per_tok

    # histogram over all logits to see distribution
    logs["router/logits"] = wandb.Histogram(
        router_logits.type(torch.float32).flatten().cpu().numpy()
    )

    # distribution over experts for layer 0, layer n/2, n-1
    for layer in [0, n_layers // 2, n_layers - 1]:
        router_logits_layer = router_logits[:, layer]
        # shape [batches, batch_size * sequence_length, num_experts_per_tok]
        weights, selected_experts = torch.topk(
            router_logits_layer, num_experts_per_tok, dim=-1
        )
        # shape [batches, batch_size * sequence_length, num_experts_per_tok, num_experts]
        expert_mask = F.one_hot(selected_experts, num_experts)
        # For a given token, determine if it was routed to a given expert.
        # Shape: [batches, batch_size * sequence_length, num_experts]
        expert_mask, _ = torch.max(expert_mask, dim=-2)
        # shape [num_experts]
        tokens_per_expert = torch.mean(expert_mask, dim=(0, 1), dtype=torch.float32)
        layer_token_routing = {
            f"router/layer_{layer}_expert_{i}_selection": tokens_per_expert[i].item()
            for i in range(num_experts)
        }
        logs.update(layer_token_routing)
    return logs


def compute_maxvio(selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute the MaxVio metric for a batch.

    Args:
        selected_experts (Tensor): Tensor of shape [num_tokens, top_k] containing
                                   the indices of the selected experts per token.
        num_experts (int): The number of experts (e.g., non-shared experts).

    Returns:
        Tensor: The maximal violation (MaxVio) for the current batch.
    """
    # Flatten selected experts (if top_k > 1)
    selected = selected_experts.view(-1)
    #load_counts are as follows : 0th idx shows the load of expert id 0 and so on.
    load_counts = torch.bincount(selected, minlength=num_experts).float() # Count tokens assigned to each expert.
    #print("Load counts per expert:", load_counts)
    
    total_tokens = selected.numel() # Total number of tokens processed.
    
    expected_load = total_tokens / num_experts     # Expected load under perfect balance.
        
    max_load = load_counts.max()    
    maxvio = (max_load - expected_load) / expected_load    
    return maxvio, load_counts  #return load counts for ratio calc