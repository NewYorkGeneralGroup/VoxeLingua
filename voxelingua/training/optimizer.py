from typing import Iterator, Tuple
import torch
from torch.optim import Optimizer
import math

class AdaptiveOptimizer(Optimizer):
    """
    Advanced optimizer with adaptive learning rates and momentum
    """
    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        relative_step: bool = True,
        scale_parameter: bool = True,
        warmup_init: bool = True
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            relative_step=relative_step,
            scale_parameter=scale_parameter,
            warmup_init=warmup_init
        )
        super().__init__(params, defaults)
        
        # Initialize buffers for adaptive learning rates
        self.base_lrs = []
        self.parameter_sizes = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.base_lrs.append(group['lr'])
                    self.parameter_sizes.append(p.numel())
                    
            state = self.state[p]
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)
            
            if warmup_init:
                state['lr'] = 0.0
            else:
                state['lr'] = group['lr']
                
    def _get_scaling_factor(self, state: dict, group: dict) -> float:
        """Calculate adaptive scaling factor"""
        step = state['step']
        
        if group['relative_step']:
            lr = min(1.0, math.sqrt(step + 1) / math.sqrt(10000))
        else:
            lr = 1.0
            
        if group['scale_parameter']:
            lr = lr / math.sqrt(self.parameter_sizes[0])
            
        return lr
        
    def _get_adaptive_lr(self, p: torch.Tensor, group: dict, state: dict) -> float:
        """Calculate adaptive learning rate for parameter"""
        beta1, beta2 = group['betas']
        step = state['step']
        
        # Compute bias correction terms
        bias_correction1 = 1 - beta1 ** (step + 1)
        bias_correction2 = 1 - beta2 ** (step + 1)
        
        # Get parameter-specific learning rate
        if group['relative_step']:
            lr = self._get_scaling_factor(state, group)
        else:
            lr = group['lr']
            
        # Apply warmup
        if group['warmup_init']:
            lr = lr * min(1.0, step / 1000)
            
        return lr
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with adaptive updates"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # Handle sparse gradients
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported')
                    
                state = self.state[p]
                state['step'] += 1
                
                # Get adaptive learning rate
                lr = self._get_adaptive_lr(p, group, state)
                
                # Compute momentum terms
                beta1, beta2 = group['betas']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Update momentum terms
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute denominator
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Update parameters
                step_size = lr * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
                
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * lr)
                    
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss
