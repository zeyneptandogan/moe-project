import torch
from torch.optim.optimizer import Optimizer

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union


from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

def _matrix_power(
    matrix: torch.Tensor,
    power: float,
    use_trunc: bool = False,
    max_rank: Optional[int]   = None,
    rank_ratio: Optional[float] = None,
    niter: int                   = 2,
) :
    """
    Raise `matrix` to the given `power` either via full SVD or truncated SVD.

    Args:
        matrix: (d×d) tensor to decompose (symmetric PD).
        power: exponent to apply to singular values.
        use_trunc: if True, use truncated SVD (torch.svd_lowrank).
        rank: number of top singular modes to keep (only if use_trunc=True).
              Defaults to full rank.
        niter: number of power‐iteration steps for svd_lowrank.

    Returns:
        matrix^power on the original device.
    """
    device = matrix.device
    # move to CPU for decomposition
    P = matrix.to('cpu')

    device = matrix.device
    P = matrix.to("cpu")
    d = P.size(0)

    if use_trunc:
        # decide rank per-dimension
        if max_rank is not None:
            k = min(d, max_rank)
        elif rank_ratio is not None:
            # e.g. rank_ratio = 0.1 to keep top-10%
            k = max(1, int(rank_ratio * d))
        else:
            # fallback: full rank
            k = d

        # truncated SVD
        U, S, V = torch.svd_lowrank(P, q=k, niter=niter)
        S = S.pow(power)
        P_power = U @ torch.diag(S) @ V.t()

    else:
        u, s, v = torch.svd(P)
        P_power = u @ s.pow(power).diag() @ v.t()

    return P_power.to(device)

def _matrix_power_qr(
    P: torch.Tensor,
    Q_prev: torch.Tensor,
    power: float,
    eps: float = 1e-6
):
    """
    Approximate P^power via one power‐iteration + QR.

    Args:
      P:   (d×d) symmetric PD preconditioner (on CPU)
      Q_prev: (d×d) last basis (orthonormal)
      power: desired exponent, e.g. -1/n
      eps: jitter on the diag
    
    Returns:
      P_power ≈ P^power (d×d),
      Q_new     (d×d) orthonormal basis for next iteration
    """
    # 1) jitter & symmetrize
    d = P.size(0)
    P = P + eps * torch.eye(d, device=P.device, dtype=P.dtype)
    P = 0.5 * (P + P.t())

    # 2) power-iteration step
    B = P @ Q_prev

    # 3) QR factorization
    Q_new, R = torch.linalg.qr(B)

    # 4) estimate eigenvalues via Rayleigh quotients
    #    λ_i ≈ (Q_new^T P Q_new)_{ii}
    lam = torch.diag(Q_new.t() @ P @ Q_new)

    # 5) form P^power ≈ Q_new diag(lam^power) Q_new^T
    lam_pow = lam.clamp(min=eps).pow(power)
    P_power = Q_new @ torch.diag(lam_pow) @ Q_new.t()

    return P_power, Q_new

class Shampoo(Optimizer):
    r"""Implements Shampoo Optimizer Algorithm.

    It has been proposed in `Shampoo: Preconditioned Stochastic Tensor
    Optimization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: epsilon added to each mat_gbar_j for numerical stability
            (default: 1e-4)
        update_freq: update frequency to compute inverse (default: 1)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Shampoo(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1802.09568

    Note:
        Reference code: https://github.com/moskomule/shampoo.pytorch
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1,
        use_qr: bool = True,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if epsilon < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if update_freq < 1:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
            use_qr=use_qr,               # for qr factorization
        )
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Safely fetch hyperparameters, falling back to defaults
            lr = group.get("lr", self.defaults["lr"])
            momentum = group.get("momentum", self.defaults["momentum"])
            weight_decay = group.get("weight_decay", self.defaults["weight_decay"])
            eps = group.get("epsilon", self.defaults["epsilon"])
            update_freq = group.get("update_freq", self.defaults["update_freq"])
            use_qr = group.get("use_qr", self.defaults.get("use_qr", False))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]

                # Initialize state if empty
                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        state[f"precond_{dim_id}"] = (
                            eps * torch.eye(dim, out=grad.new(dim, dim))
                        )
                        state[f"inv_precond_{dim_id}"] = grad.new(dim, dim).zero_()
                        if use_qr:
                            state[f"basis_{dim_id}"] = torch.eye(dim, device=p.device, dtype=p.dtype)

                # Momentum and weight decay
                if momentum > 0:
                    buf = state["momentum_buffer"]
                    grad.mul_(1 - momentum).add_(buf, alpha=momentum)
                if weight_decay > 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Per-mode preconditioning
                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]

                    # Unfold gradient along this mode
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)
                    grad_t = grad.t()

                    # Update covariance
                    precond.add_(grad @ grad_t)

                    # Recompute preconditioner at specified frequency
                    if state["step"] % update_freq == 0:
                        if use_qr:
                            # QR-based path
                            P_cpu = precond.to("cpu")
                            Q_prev = state[f"basis_{dim_id}"].to("cpu")
                            P_pow, Q_new = _matrix_power_qr(
                                P_cpu, power=-1.0/order, Q_prev=Q_prev, eps=eps
                            )
                            state[f"basis_{dim_id}"] = Q_new.to(p.device)
                            inv_precond.copy_(P_pow.to(p.device))
                        else:
                            # SVD-based path (truncated or full)
                            inv_precond.copy_(
                                _matrix_power(
                                    precond,
                                    -1.0 / order,
                                    use_trunc=group.get("use_trunc", False),
                                    max_rank=group.get("max_rank", None),
                                    rank_ratio=group.get("rank_ratio", None),
                                    niter=group.get("niter", 2),
                                )
                            )

                    # Apply preconditioner
                    if dim_id == order - 1:
                        grad = (grad_t @ inv_precond).view(original_size)
                    else:
                        grad = (inv_precond @ grad).view(transposed_size)

                # Finalize step
                state["step"] += 1
                if momentum > 0:
                    state["momentum_buffer"] = grad
                p.data.add_(grad, alpha=-lr)

        return loss
