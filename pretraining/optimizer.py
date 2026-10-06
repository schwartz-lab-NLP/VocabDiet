import os
import sys
import torch
from torch import Tensor, nn
import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


# -----------------------------------------------------------------------------
# Muon optimizer


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.

        if not is_distributed():
            # Single GPU implementation
            for group in self.param_groups:
                params = group["params"]
                momentum = group["momentum"]
                for p in params:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    eff_lr = (
                        group["lr"]
                        * max(1, p.size(-2) / p.size(-1)) ** 0.5
                        * getattr(p, "lr_mul", 1.0)
                    )
                    eff_weight_decay = (
                        group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    )
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                    p.add_(other=v, alpha=-eff_lr)
            return

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [
                torch.zeros_like(params[-1])
            ] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(
                    dist.reduce_scatter(
                        grad,
                        grad_pad[base_i : base_i + world_size],
                        op=dist.ReduceOp.AVG,
                        async_op=True,
                    ).get_future()
                )

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = (
                        group["lr"]
                        * max(1, p.size(-2) / p.size(-1)) ** 0.5
                        * getattr(p, "lr_mul", 1.0)
                    )
                    eff_weight_decay = (
                        group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    )
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_reduce_futures.append(
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                        async_op=True,
                    ).get_future()
                )
        torch.futures.collect_all(all_reduce_futures).wait()


class DistAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        use_all_reduce_for_small_params: bool = False,
        use_all_reduce_for_all_params: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_all_reduce_for_small_params=use_all_reduce_for_small_params,
            use_all_reduce_for_all_params=use_all_reduce_for_all_params,
        )
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal

    @torch.compile
    @torch.no_grad()
    def step(self):
        if not is_distributed():
            # Single GPU implementation - standard Adam
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]
                params = group["params"]
                for param in params:
                    if param.grad is None:
                        continue

                    grad = param.grad
                    lr = group["lr"] * getattr(param, "lr_mul", 1.0)
                    state = self.state[param]

                    # State init
                    if not state:
                        state["step"] = torch.tensor(0, dtype=torch.int64, device=param.device)
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    t = state["step"]

                    # weight decay
                    if wd != 0:
                        eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                        param.mul_(1 - eff_weight_decay)

                    # update running averages
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # bias corrections
                    bias1 = 1 - beta1**t
                    bias2 = 1 - beta2**t

                    # compute step
                    denom = exp_avg_sq.sqrt().add_(eps)
                    step_size = lr * (torch.sqrt(bias2) / bias1)
                    update = exp_avg.div(denom).mul_(step_size)
                    param.add_(other=update, alpha=-1.0)
            return

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Check which distributed strategy to use
        use_all_reduce = any(
            group.get("use_all_reduce_for_all_params", False) for group in self.param_groups
        )
        use_hybrid = any(
            group.get("use_all_reduce_for_small_params", False) for group in self.param_groups
        )

        if use_all_reduce:
            # Simple all_reduce for all parameters (no sharding, for testing/debugging)
            all_reduce_futures = []
            for group in self.param_groups:
                params = group["params"]
                for param in params:
                    if param.grad is None:
                        continue
                    # All-reduce gradient across all ranks
                    all_reduce_futures.append(
                        dist.all_reduce(
                            param.grad, op=dist.ReduceOp.AVG, async_op=True
                        ).get_future()
                    )

            # Update all parameters with standard Adam (no sharding)
            idx = 0
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]
                params = group["params"]
                for param in params:
                    if param.grad is None:
                        continue

                    all_reduce_futures[idx].wait()
                    idx += 1

                    lr = group["lr"] * getattr(param, "lr_mul", 1.0)
                    grad = param.grad
                    state = self.state[param]

                    # State init
                    if not state:
                        state["step"] = torch.tensor(0, dtype=torch.int64, device=param.device)
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    t = state["step"]

                    # weight decay
                    if wd != 0:
                        eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                        param.mul_(1 - eff_weight_decay)

                    # update running averages
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # bias corrections
                    bias1 = 1 - beta1**t
                    bias2 = 1 - beta2**t

                    # compute step
                    denom = exp_avg_sq.sqrt().add_(eps)
                    step_size = lr * (torch.sqrt(bias2) / bias1)
                    update = exp_avg.div(denom).mul_(step_size)
                    param.add_(other=update, alpha=-1.0)
        elif use_hybrid:
            # Hybrid implementation: small params use all_reduce, large params use reduce_scatter
            # Classify parameters by size
            small_params_list = []  # (group, param) tuples for params with shape[0] < world_size
            large_params_list = []  # (group, param) tuples for params with shape[0] >= world_size

            for group in self.param_groups:
                params = group["params"]
                for param in params:
                    if param.grad is None:
                        continue
                    if param.shape[0] < world_size:
                        small_params_list.append((group, param))
                    else:
                        large_params_list.append((group, param))

            # Process small parameters with all_reduce (no sharding)
            small_all_reduce_futures = []
            for group, param in small_params_list:
                # All-reduce gradient across all ranks
                small_all_reduce_futures.append(
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                )

            # Process large parameters with reduce_scatter (sharded updates)
            reduce_scatter_futures: list[torch.Future] = []
            grad_slices = []
            for group, param in large_params_list:
                grad = param.grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(
                    dist.reduce_scatter_tensor(
                        grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                    ).get_future()
                )
                grad_slices.append(grad_slice)

            # Update small parameters (standard Adam, no sharding)
            for i, (group, param) in enumerate(small_params_list):
                small_all_reduce_futures[i].wait()
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]
                lr = group["lr"] * getattr(param, "lr_mul", 1.0)

                grad = param.grad
                state = self.state[param]

                # State init
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int64, device=param.device)
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    param.mul_(1 - eff_weight_decay)

                # update running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias corrections
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t

                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                param.add_(other=update, alpha=-1.0)

            # Update large parameters (sharded Adam)
            all_gather_futures = []
            for i, (group, param) in enumerate(large_params_list):
                reduce_scatter_futures[i].wait()
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]
                lr = group["lr"] * getattr(param, "lr_mul", 1.0)

                rank_size = param.shape[0] // world_size
                p_slice = param[rank * rank_size : (rank + 1) * rank_size]
                g_slice = grad_slices[i]
                state = self.state[param]

                # State init
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int64, device=param.device)
                    state["exp_avg"] = torch.zeros_like(p_slice)
                    state["exp_avg_sq"] = torch.zeros_like(p_slice)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)

                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)

                # bias corrections
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t

                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)

                all_gather_futures.append(
                    dist.all_gather_into_tensor(param, p_slice, async_op=True).get_future()
                )

            # Wait for all all_gather operations to complete
            torch.futures.collect_all(all_gather_futures).wait()
        else:
            # Original implementation: all parameters use reduce_scatter_tensor
            reduce_scatter_futures: list[torch.Future] = []
            all_reduce_futures: list[torch.Future] = []
            grad_slices = []
            for group in self.param_groups:
                params: list[Tensor] = group["params"]
                grad = torch.empty_like(params[-1])
                for base_i in range(len(params)):
                    param = params[base_i]
                    # Skip parameters without gradients
                    if param.grad is None:
                        continue
                    grad = param.grad
                    rank_size = grad.shape[0] // world_size
                    grad_slice = torch.empty_like(grad[:rank_size])

                    reduce_scatter_futures.append(
                        dist.reduce_scatter_tensor(
                            grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                        ).get_future()
                    )
                    grad_slices.append(grad_slice)

            idx = 0
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]
                params = group["params"]
                for base in range(len(params)):
                    param = params[base]
                    # Skip parameters without gradients
                    if param.grad is None:
                        continue

                    reduce_scatter_futures[idx].wait()
                    rank_size = param.shape[0] // world_size
                    p_slice = param[rank * rank_size : (rank + 1) * rank_size]
                    lr = group["lr"] * getattr(param, "lr_mul", 1.0)
                    state = self.state[param]
                    g_slice = grad_slices[idx]
                    # State init
                    if not state:
                        state["step"] = torch.tensor(0, dtype=torch.int64, device=param.device)
                        state["exp_avg"] = torch.zeros_like(p_slice)
                        state["exp_avg_sq"] = torch.zeros_like(p_slice)
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    t = state["step"]
                    # weight decay
                    if wd != 0:
                        eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                        p_slice.mul_(1 - eff_weight_decay)
                    # update running averages
                    exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                    # bias corrections
                    bias1 = 1 - beta1**t
                    bias2 = 1 - beta2**t
                    # compute step
                    denom = exp_avg_sq.sqrt().add_(eps)
                    step_size = lr * (torch.sqrt(bias2) / bias1)
                    update = exp_avg.div(denom).mul_(step_size)
                    p_slice.add_(other=update, alpha=-1.0)
                    idx += 1
                    all_reduce_futures.append(
                        dist.all_gather_into_tensor(param, p_slice, async_op=True).get_future()
                    )
            torch.futures.collect_all(all_reduce_futures).wait()
