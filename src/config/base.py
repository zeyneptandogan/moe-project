import distributed


def parse_args(base_parser, args, namespace):
    parser = base_parser
    # General training params
    parser.add_argument("--experiment-name", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--data-seed", default=1337, type=int)
    parser.add_argument("--eval-interval", default=200, type=int)
    parser.add_argument("--full-eval-at", nargs="+", type=int)
    parser.add_argument("--eval-batches", default=32, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument(
        "--distributed-backend",
        default=None,
        type=str,
        required=False,
        choices=distributed.registered_backends(),
    )
    parser.add_argument("--log-interval", default=50, type=int)

    # Checkpointing
    parser.add_argument("--results-base-folder", default="./exps", type=str)
    parser.add_argument("--permanent-ckpt-interval", default=0, type=int)
    parser.add_argument("--latest-ckpt-interval", default=0, type=int)
    parser.add_argument("--resume-from", default=None, type=str)
    parser.add_argument("--resume-from-swa", default=None, type=str)

    parser.add_argument("--auto-resume", default=True)

    # logging params (WandB)
    parser.add_argument("--wandb", action="store_true")  # whether to use wandb or not
    parser.add_argument("--wandb-project", default="my-project", type=str)
    parser.add_argument(
        "--wandb-run-prefix", default="none", type=str
    )  # is added before the autogenerated experiment name
    parser.add_argument(
        "--eval-seq-prefix", default="none", type=str
    )  # prefix used to generate sequences
    parser.add_argument("--log-dynamics", action="store_true")
    parser.add_argument(
        "--dynamics-logger-cfg", default="./src/logger/rotational_logger.yaml", type=str
    )

    # Schedule
    parser.add_argument(
        "--scheduler",
        default="cos",
        choices=["linear", "cos", "wsd", "none", "cos_inf"],
    )
    parser.add_argument("--cos-inf-steps", default=0, type=int)
    # parser.add_argument("--cos-final-lr", default=1e-6, type=float)
    parser.add_argument("--iterations", default=15000, type=int)
    parser.add_argument("--warmup-steps", default=300, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--min-lr", type=float) # for cosine schedule
    # wsd
    parser.add_argument("--wsd-final-lr-scale", default=0.0, type=float)
    parser.add_argument("--wsd-fract-decay", default=0.1, type=float)
    #parser.add_argument("--wsd-exponential-decay", action="store_true")
    parser.add_argument("--decay-type",default="linear",choices=["linear","cosine","exp","miror_cosine","square","sqrt"])
    # Optimization
    parser.add_argument("--opt", default="adamw", choices=["adamw", "sgd","SFAdamW", "Shampoo"])
    parser.add_argument("--batch-size", default=50, type=int)
    parser.add_argument("--acc-steps", default=4, type=int)
    parser.add_argument("--weight-decay", default=1e-1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument(
        "--grad-clip", default=1.0, type=float
    )  # default value is 1.0 in NanoGPT

    # Weight Averaging
    parser.add_argument("--weight-average", action="store_true")
    parser.add_argument(
        "--wa-interval",
        default=5,
        type=int,
        help="How often to take the average (every k steps). Must divide wa-horizon.",
    )
    parser.add_argument(
        "--wa-horizon",
        default=500,
        type=int,
        help="How frequently we save uniform model averages. Should divide "
        + "latest-ckpt-interval, otherwise some points may not be saved "
        + "correctly.",
    )
    parser.add_argument(
        "--wa-dtype",
        default="float32",
        type=str,
        choices=["float32", "float64"],
    )

    parser.add_argument("--wa-use-temp-dir", action="store_true")
    parser.add_argument("--wa-sweep-horizon", action="store_true")
    parser.add_argument("--max-num-wa-sweeps", default=5, type=int)

    parser.add_argument("--exponential-moving-average", action="store_true")
    parser.add_argument(
        "--ema-interval",
        default=10,
        type=int,
        help="How often to take the EMA average (every k steps).",
    )
    parser.add_argument(
        "--ema-decay",
        default=0.95,
        type=float,
        help="EMA decay parameter (between 0.9 and 1).",
    )
    parser.add_argument(
        "--ema-after-warmup",
        action="store_true",
        help="Start EMA after warmup steps.",
    )

    # Dataset params
    parser.add_argument("--datasets-dir", type=str, default="./datasets/")
    parser.add_argument(
        "--dataset",
        default="fineweb",
        choices=[
            "wikitext",
            "shakespeare-char",
            "arxiv",
            "arxiv2000",
            "arxiv+wiki",
            "openwebtext2",
            "redpajama",
            "slimpajama",
            "slimpajama_chunk1",
            "redpajamav2",
            "fineweb",
        ],
    )
    parser.add_argument(
        "--tokenizer", default="gpt2", type=str, choices=["gpt2", "mistral"]
    )
    parser.add_argument("--vocab-size", default=50304, type=int)
    parser.add_argument(
        "--data-in-ram", action="store_true"
    )  # force the data to RAM, mostly useless except for openwebtext2

    # Model params
    parser.add_argument(
        "--model",
        default="llama",
        choices=[
            "base",
            "llama"
        ],
    )
    parser.add_argument("--parallel-block", action="store_true")
    parser.add_argument(
        "--use-pretrained", default="none", type=str
    )  # 'none', 'gpt-2' or a path to the pretraind model
    parser.add_argument("--from-dense", action="store_true")
    parser.add_argument("--init-std", default=0.02, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--n-head", default=12, type=int)
    parser.add_argument("--n-layer", default=24, type=int)  # depths in att + ff blocks
    parser.add_argument("--sequence-length", default=512, type=int)
    parser.add_argument(
        "--n-embd", default=768, type=int  # embedding size / hidden size ...
    )
    parser.add_argument(
        "--multiple-of",  # make SwiGLU hidden layer size multiple of large power of 2
        default=256,
        type=int,
    )
    parser.add_argument("--rmsnorm-eps", default=1e-5, type=float)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        type=str,
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--bias", default=False, type=bool)
    parser.add_argument("--compile", action="store_true")
    ### moe params
    parser.add_argument("--moe", action="store_true")
    parser.add_argument(
        "--moe-routing",
        default="standard_gating",
        type=str,
        choices=["standard_gating", "expert_choice"],
    )
    parser.add_argument("--moe-num-experts", default=8, type=int)
    parser.add_argument(  # only used for expert choice routing
        "--capacity-factor", default=2.0, type=float
    )
    parser.add_argument(  # deepseek routing, experts that are always active
        "--moe-num-shared-experts", default=0, type=int
    )
    parser.add_argument(
        "--moe-router-loss",
        default="load_balancing_z_loss",
        type=str,
        choices=["entropy", "load_balancing_only", "load_balancing_z_loss"],
    )
    parser.add_argument("--moe-num-experts-per-tok", default=2, type=int)
    parser.add_argument("--moe-entropy-loss-factor", default=0.01, type=float)
    parser.add_argument("--moe-aux-loss-factor", default=0.1, type=float)
    parser.add_argument("--moe-z-loss-factor", default=0.01, type=float)
    parser.add_argument(
        "--moe-softmax-order",
        type=str,
        default="topk_softmax",
        choices=["softmax_topk", "topk_softmax"],
    )
    parser.add_argument("--plot-router-logits", action="store_true")
    parser.add_argument("--mlp-dim-exp-factor", default=1.0, type=float)
    ###
    ## mup arguments
    parser.add_argument("--scale-emb", default=10, type=int)
    # the base model width that mup has been configured on
    parser.add_argument("--scale-base-model", default=256, type=int)
    parser.add_argument("--scale-depth", default=1.4, type=float)
    parser.add_argument("--expert_lr", default=2e-3, type=float)
    # loss free aux method
    parser.add_argument("--aux-loss-free", action="store_true")
    parser.add_argument("--bias-update-rate", default=1e-3, type=float)
    
    #for shampoo optimizer
    parser.add_argument("--eps",          type=float, default=1e-8,
                    help="Shampoo numerical‑stability epsilon")
    parser.add_argument("--shampoo_decay",type=float, default=0.9,
                        help="EMA decay rate for Shampoo covariances")
    
    #for load based lr update
    parser.add_argument("--ratio-update-lr", action="store_true")
    return parser.parse_args(args, namespace)