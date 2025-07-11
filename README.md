# Codebase: Finding the Right Optimization for Mixture-of-Experts
This repository accompanies the semester project **Finding the Right Optimization for Mixture-of-Experts**, authored by Zeynep Tandogan, Alexander Hägele, and Prof. Martin Jaggi from EPFL.


## Abstract
Mixture‐of‐Experts (MoE) models offer massive capacity at reduced compute by routing each input to a subset of “experts,” but in practice they suffer from severe load imbalance, expressivity loss in infrequently activated experts, and a trade‐off between strict balancing and overall performance. In this work, we perform a comprehensive study of MoE optimization techniques, including:

1. Differentiated learning-rate schedules for expert vs. non-expert parameters.  
2. Systematic tuning of auxiliary loss coefficients along with loss-free balancing methods.  
3. Comparison of optimizers (e.g., AdamW vs. Shampoo) and activation functions (sigmoid vs. softmax).  
4. Varying the number of experts to assess capacity vs. utilization trade-offs.  

Our experiments reveal that certain configurations yield improvements in validation loss, perplexity, and accuracy, while other hyperparameter choices can lead to modest degradations. These findings inform best practices for training efficient and balanced MoE systems.


## Key Contributions
- **Auxiliary Loss vs. Loss-Free Balancing**: Ablation of tuning coefficients and evaluation of loss-free methods for expert load balance.  
- **Expert-Specific LR Schedules**: Implementation of static and load-adaptive learning rates for expert and non-expert parameters.  
- **Optimizer Comparison**: Benchmarking AdamW against Shampoo to understand their effects on expert utilization and convergence.  
- **Scalability Analysis**: Investigation of the impact of varying the number of experts on model performance and training efficiency.


## Quickstart 

Create a conda environment and install dependencies (we recommend Python 3.10):

```bash
conda create -n moe-env python=3.12.9 -y
conda activate moe-env
pip install -r requirements.txt
```

Run a simple training on the Fineweb EDU dataset:
```bash
python ./src/main.py
```

You can see the default parameters and additional parameters in src/config/base.py.

Sample Run (including aux loss free method with load based LR update)

```bash
torchrun --nproc-per-node=1 src/main.py \
  --wandb-project run_moe \
  --dataset fineweb \
  --distributed-backend nccl \
  --latest-ckpt-interval 1000 \
  --model llama \
  --compile \
  --lr 0.0001 \
  --expert_lr 0.00001 \
  --iterations 2000 \
  --moe \
  --plot-router-logits \
  --batch-size 5 \
  --acc-steps 4 \
  --aux-loss-free \
  --ratio-update-lr
```

#### Parameter Details (important flags for the experiments)
- `--lr`: Base learning rate for non-expert parameters.  
- `--expert_lr`: Learning rate for expert parameters.  
- `--opt [adamw|sgd|SFAdamW|Shampoo]`: Choice of optimizer (default: adamw).  
- `--eps`: Shampoo numerical-stability epsilon (default: 1e-8).  
- `--shampoo_decay`: EMA decay rate for Shampoo covariances (default: 0.9).  
- `--aux-loss-free`: Enable the loss-free load balancing method (no auxiliary loss).  
- `--bias-update-rate`: Router bias learning rate when using loss-free balancing (default: 1e-3).  
- `--moe-aux-loss-factor`: Weight of the auxiliary balancing loss in MoE routing (default: 0.1).  
- `--ratio-update-lr`: Turn on load-adaptive LR updates, scaling each expert’s LR by its usage ratio.

## Reports & Presentation
You can find the project deliverables in the repo:

- [MoE Final Report (PDF)](zeyneptandogan_moe_report.pdf)  
- [MoE Presentation (PDF)](zeyneptandogan_moe_presentation.pdf)  
