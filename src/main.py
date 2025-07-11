import argparse
import json
from pathlib import Path
import random
import os 
import schedulefree
from distributed.shampoo import Shampoo
import numpy as np
import torch
import wandb
from collections import defaultdict


import config
from data.utils import DataReader, get_dataset
import distributed
from models.utils import get_model
from optim.base import train
from optim.utils import cos_inf_schedule, wsd_schedule
from optimizers.distributed_shampoo import DistributedShampoo, AdamGraftingConfig

# Suppress warnings from torch.distributed
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

def main(args):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []
    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8

    if args.moe and args.ratio_update_lr:
        print('***LOAD BASED LR UPDATE FOR EXPERTS***')

    exp_name = get_exp_name(args, distributed_backend)
    exp_dir = Path(args.results_base_folder) / exp_name

    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if not args.auto_resume:
            raise ValueError(
                f"The experiment dir {exp_dir} already exists. "
                + "To resume training, set auto_resume=True. "
                + "Otherwise, specify a different experiment name. "
            )
        else:
            # Auto resume overwrites resume_from
            args.resume_from = str(exp_dir / "ckpts" / "latest")

    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

    wandb_run_id = None
    if distributed_backend.is_master_process() and args.wandb:
        if args.resume_from:
            wandb_run_id_file = Path(args.resume_from) / "wandb_id.txt" # instead of storing in the checkpoint folder, store in a txt so that we wont reload the checkpoint twice.
            if wandb_run_id_file.exists():
                with open(wandb_run_id_file, "r") as f:
                    wandb_run_id = f.read().strip()

        if wandb_run_id:
            wandb.init(
                project=args.wandb_project,
                name=exp_name,
                config=vars(args),
                entity="Moe-tmp",  
                id=wandb_run_id,
                resume="allow"  
            )
        else:
            wandb.init(
                project=args.wandb_project,
                name=exp_name,
                config=vars(args),
                entity="Moe-tmp",
            )
        #print(wandb.run.id)
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("lr", step_metric="iter")
        print(f"Logging to wandb as {wandb.run.name}")
        print("W&B Run URL:", wandb.run.get_url())

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    model = get_model(args).to(args.device)
    # TODO: take care of initializing the model if args.use_pretrained != 'none'
    print(f"\nModel:\n{model}")

    model = distributed_backend.transform_model(model)
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0

    if args.moe and not args.ratio_update_lr:
        updated_group_specs = []
        for spec in group_specs:
            mlp_found = False
            # Initialize once per spec so values can be accumulated
            mlp_param = {
                "params": [],
                "lr": args.expert_lr,
                "group": "expert",
            }
            other_param = {
                "params": [],
                "lr": args.lr,
                "group": "global",  #non-expert
            }
            for param in spec.get('params', []):
                if '.mlp.experts.' in param: # or '.mlp.router.' in param:
                    mlp_param["params"].append(param)
                    mlp_found = True
                else:
                    other_param["params"].append(param)
            if mlp_found:
                updated_group_specs.append(mlp_param)
                updated_group_specs.append(other_param)
                #print("other params:", other_param)
            else:
                updated_group_specs.append(spec)
        group_specs = updated_group_specs

    
            
    if args.moe and args.ratio_update_lr:
        updated_group_specs = []

        num_experts          = args.moe_num_experts        
        expert_lr_base       = args.expert_lr                
        global_lr            = args.lr

        # 1.  Prepare an empty list of lists, one slot per expert
        expert_param_dict = defaultdict(list)    # key: (layer_id, expert_id)
        router_params      = []      
        global_params      = []      

        # 2.  Traverse *all* parameters in the existing specs
        for spec in group_specs:
            for param in spec.get("params", []): 
                if ".mlp.experts." in param:
                    #  "...mlp.experts.<id>."
                    layer_id = int(param.split("transformer.h.")[1].split(".")[0])
                    expert_id = int(param.split(".mlp.experts.")[1].split(".")[0])
                    expert_param_dict[(layer_id, expert_id)].append(param)
                elif ".mlp.router." in param:
                    router_params.append(param)
                else:
                    global_params.append(param)
            
        num_expert_groups = len(expert_param_dict)
        num_expert_params = sum(len(plist) for plist in expert_param_dict.values())
        print(f"[DEBUG] {num_expert_groups} expert groups, total {num_expert_params} expert parameters")

        # 3.  One group per expert
        for (layer_id, expert_id), plist in expert_param_dict.items():
            updated_group_specs.append(
                {
                    "params": plist,
                    "lr":     expert_lr_base,
                    "group":  "expert",
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                }
            )

        if router_params:
            updated_group_specs.append({
                "params": router_params,
                "lr":     global_lr, #expert_lr_base,      
                "group":  "router"
            })

        # 4.  The global / non-expert group
        updated_group_specs.append({
            "params": global_params,
            "lr":     global_lr,
            "group":  "global"
        })
        group_specs = updated_group_specs

    for i, spec in enumerate(group_specs):
        grp = spec.get("group", "<no group>")
        lr  = spec.get("lr",     None)
        n   = len(spec.get("params", []))

        if isinstance(lr, (float, int)):
            lr_str = f"{lr:.2e}"
        else:
            lr_str = str(lr)

        print(f"Group #{i:2d}: name={grp:8s}  lr={lr_str:>8s}  #params={n}")

    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
            #print("Original name:", p_name, "translated to:", translated_p_names)  # Debug print
            for tp in translated_p_names:
                if tp not in param_name_mapping:
                    print("Missing key in param_name_mapping:", tp)  # Debug print for missing key
                else:
                    params.append(param_name_mapping[tp])
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
  
    if args.wandb and distributed_backend.is_master_process():
        wandb.log(
            {"parameters": params_cnt, "optimized_parameters": optimized_params_cnt}
        )

    if args.opt == "adamw":
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr, #global lr
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "SFAdamW":
        opt = schedulefree.AdamWScheduleFree(
            group_specs, 
            lr=args.lr, #global lr
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
        )
    elif args.opt == "Shampoo":
        # Shampoo optimizer - It requires the learning rate, momentum, weight_decay, epsilon and update frequency.

        opt = DistributedShampoo(
            group_specs,                       # your param groups
            lr=args.lr,                        # same base LR as before
            betas=(args.beta1, args.beta2),    # EMA for Shampoo’s accumulators
            epsilon=args.eps,                  # Shampoo’s regularizer (match your old eps)
            weight_decay=args.weight_decay,    # decoupled (AdamW‑style) decay
            use_decoupled_weight_decay=True,

            # graft onto AdamW’s diagonal update:
            grafting_config=AdamGraftingConfig(
                beta2=args.beta2,              # match your old AdamW beta2
                epsilon=args.eps,              # match your old AdamW eps
            ),

            # Shampoo performance knobs (tune or expose via args as needed):
            max_preconditioner_dim=getattr(args, "max_preconditioner_dim", 768),
            precondition_frequency=getattr(args, "precondition_frequency", 100),
        )
    else:
        opt = torch.optim.SGD(
            group_specs, 
            lr=args.lr,  #global lr
            momentum=0.9, 
            weight_decay=args.weight_decay
        )
    print(f"\nOptimizer:\n{opt}")

    if args.scheduler != "none":
        assert args.warmup_steps < args.iterations, "Warmup steps must be < iterations."

        div_factor = 1e2
        if hasattr(args, "min_lr") and args.min_lr is not None:
            final_div_factor = (args.lr / div_factor) / args.min_lr # to be used for scheduler cos, linear and cos_inf
        else:
            # Otherwise, use a default value (e.g. 0.1)
            final_div_factor = 0.1

        if args.scheduler in ["cos", "linear"]:
            # initial lr is args.lr / div_factor
            # final lr is initial_lr/final_div_factor = args.lr / div_factor / final_div_factor
            # TODO: use this argument
            # final_div_factor = args.lr / 1e2 / args.cos_final_lr
            # to have final lr = min lr -> we should set final_div_factor = args.lr / div_factor / min_lr
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt,
                max_lr=[group.get("lr", args.lr) for group in group_specs],
                total_steps=args.iterations,
                pct_start=args.warmup_steps / args.iterations,
                anneal_strategy=args.scheduler,
                cycle_momentum=False,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
            )
        elif args.scheduler == "cos_inf":
            lambda_schedule = cos_inf_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                n_inf=args.cos_inf_steps,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        elif args.scheduler == "wsd":
            lambda_schedule = wsd_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                fract_decay=args.wsd_fract_decay,
                init_div_factor=1e2,
                final_lr_factor=args.wsd_final_lr_scale, #should be 0 here
                decay_type=args.decay_type,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None


    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


def get_exp_name(args, distributed_backend):
    """Returns the name of the experiment, used for saving models and wandb."""
    if args.experiment_name is not None:
        return args.experiment_name

    model_prefix = "moe_" if args.moe else ""  #newly added for moe

    rank = distributed_backend.rank
    parts = [
    args.dataset,
        f"{model_prefix}{args.model}",
        f"nlayers{args.n_layer}",
        f"nhead{args.n_head}",
    ]
    
    if args.moe:
        parts.append(f"expert_lr{args.expert_lr}") #moe lr in the project name
        parts.append(f"aux{args.moe_aux_loss_factor}")

    parts.extend([
        f"lr{args.lr}",
        f"sched_{args.scheduler}",
        f"warmup{args.warmup_steps}",
        f"decay_{args.decay_type}_{args.wsd_fract_decay}",
        f"iter{args.iterations}",
        f"bs{args.batch_size}x{args.acc_steps}",
        f"ws{args.world_size}",
    ])

    exp_name = "_".join(parts)
   
    if args.wandb_run_prefix != "none":
        exp_name = args.wandb_run_prefix + "_" + exp_name
    exp_name += f"_seed{args.seed - rank}"
    exp_name += f"_data_seed{args.data_seed}"

    if args.weight_average:
        exp_name+= f"_WA"
    if args.opt == "SFAdamW":
        exp_name+= f"_beta1_{args.beta1}"
        exp_name+= f"_beta2_{args.beta2}"
    return exp_name


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        #keep_in_ram=args.data_in_ram,
        keep_in_ram=True,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
    }


if __name__ == "__main__":
    args = get_args()
    main(args)