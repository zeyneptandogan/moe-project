from contextlib import nullcontext
import copy
from pathlib import Path
import time
import yaml
from tqdm import tqdm

import torch
import wandb

from logger.logger import DynamicsLogger
from optim.weight_averaging import (
    WeightAverager,
    eval_ema,
    eval_wa,
    ExponentialWeightAverager,
)
from .utils import (
    eval,
    get_batch,
    load_checkpoint,
    load_worker_state,
    save_checkpoint,
    save_worker_state,
    visualize_routing,
    compute_maxvio
)


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)
        print(f"Model compiled.")

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not intended for fine-tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        load_worker_state(ckpt_dir)
    else:
        curr_iter = 0

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the checkpoint.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )

    if cfg.exponential_moving_average:
        ema = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ema_interval,
            decay=cfg.ema_decay,
            warmup=cfg.warmup_steps if cfg.ema_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    model.train()

    # Initialize tqdm progress bar
    progress_bar = tqdm(
        total=cfg.iterations,
        initial=curr_iter,
        desc="Training",
        position=0,
        leave=True,
        dynamic_ncols=True
    )

    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir, cfg.wandb)
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    print(f"Saving latest checkpoint at iteration {curr_iter}")
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir, cfg.wandb)
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size # number of tokens processed so far
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_and_log(
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )
            if cfg.exponential_moving_average:
                eval_ema(
                    curr_iter,
                    not_compiled_model,
                    ema,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y, moe=cfg.moe) # newly added for moe
    

            loss = outputs["loss"] / cfg.acc_steps
            loss.backward()
            substep += 1

            if cfg.moe and "selected_experts" in outputs:
                selected_experts_list = outputs["selected_experts"]
                num_non_shared_experts = cfg.moe_num_experts - cfg.moe_num_shared_experts
                maxvio_list = []
                for idx, selected_experts in enumerate(selected_experts_list):
                    #print(f"Layer {idx}: selected_experts shape: {selected_experts.shape}")
                    maxvio = compute_maxvio(selected_experts, num_non_shared_experts)
                    maxvio_list.append(maxvio)
                    #print(f"Layer {idx}: MaxViobatch = {maxvio.item():.4f}")
                
                avg_maxvio = sum(maxvio_list) / len(maxvio_list)
                #print(f"Microstep {microstep_idx}: Loss = {loss.item():.4f}, Average MaxViobatch = {avg_maxvio.item():.4f}\n")
                if cfg.wandb:
                    if distributed_backend.is_master_process():
                        wandb.log({
                            "train/maxviobatch": avg_maxvio.item(),
                            **{f"train/maxviobatch_layer_{i}": mv.item() for i, mv in enumerate(maxvio_list)}
                        })
        

        # norms logging per layer
        if cfg.wandb:
            log_dict = {}
            total_param_norm_sq = 0.0
            total_grad_norm_sq = 0.0

            for name, param in model.named_parameters():
                # Compute the L2 norm for the parameter
                param_norm = param.norm(2).item()
                total_param_norm_sq += param_norm ** 2
                log_dict[f"Param Norm/{name}"] = param_norm

                # Compute the gradient norm if the gradient exists
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()
                    total_grad_norm_sq += grad_norm ** 2
                    log_dict[f"Grad Norm/{name}"] = grad_norm

            # Calculate overall norms and log them
            overall_param_norm = total_param_norm_sq ** 0.5
            overall_grad_norm = total_grad_norm_sq ** 0.5

            log_dict["Overall Param Norm"] = overall_param_norm
            log_dict["Overall Grad Norm"] = overall_grad_norm
            if distributed_backend.is_master_process():
                wandb.log(log_dict)

        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        if cfg.weight_average:
            weight_averager.step(not_compiled_model, distributed_backend.is_master_process())
        if cfg.exponential_moving_average:
            ema.step(not_compiled_model, distributed_backend.is_master_process())
        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1
        progress_bar.update(1)  # Update tqdm progress bar

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps
            train_aux_losses = {
                f"train/{k}": v for k, v in outputs["aux_losses"].items()
            }

            global_lr = None
            expert_lr = None

            for param_group in opt.param_groups:
                if param_group.get("group") == "expert":
                    expert_lr = param_group["lr"]
                elif param_group.get("group") == "global":
                    global_lr = param_group["lr"]

            # If no expert group is found, you might set expert_lr to global_lr or skip logging it.
            if expert_lr is None:
                expert_lr = global_lr

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"global_lr={global_lr:.2e} expert_lr={expert_lr:.2e}"
            )

            if cfg.wandb:
                metrics = {
                    "iter": curr_iter,
                    "train/loss": train_loss,
                    "train/perplexity": 2.71828 ** train_loss,
                    "global_lr": global_lr,
                    "iter_dt": dt,
                    **train_aux_losses,
                }
                if expert_lr is not None:
                    metrics["expert_lr"] = expert_lr # added for moe
                wandb.log(metrics)

    progress_bar.close() 
    return stats

def eval_and_log(
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity, val_aux_losses, router_logits, avg_maxvio_global, per_layer_maxvio_global = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        moe=cfg.moe,
        get_router_logits=cfg.moe and cfg.plot_router_logits,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
                **val_aux_losses,
            }
        else:
            logs = {
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
                **val_aux_losses,
            }
        if avg_maxvio_global is not None:
            logs["val/maxvioglobal"] = avg_maxvio_global.item()
            # Log per-layer global maxvio.
            logs.update(per_layer_maxvio_global)
        if cfg.moe and cfg.plot_router_logits:
            routing_logs = visualize_routing(router_logits, cfg)
            logs = {**logs, **routing_logs}  #added for moe logging

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()

