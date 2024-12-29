from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
from tqdm import tqdm

from ..models import VoxeLinguaModel
from ..config import TrainingConfig
from .optimizer import AdaptiveOptimizer
from .scheduler import CosineWarmupScheduler

logger = logging.getLogger(__name__)

@dataclass
class TrainingState:
    """Maintains training state with checkpointing capability"""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    model_state: Optional[Dict[str, torch.Tensor]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    scaler_state: Optional[Dict[str, Any]] = None
    
    def save(self, path: Path):
        state_dict = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'model_state': self.model_state,
            'optimizer_state': self.optimizer_state,
            'scheduler_state': self.scheduler_state,
            'scaler_state': self.scaler_state
        }
        torch.save(state_dict, path)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainingState':
        state_dict = torch.load(path)
        return cls(**state_dict)

class DistributedTrainer:
    """
    Advanced distributed trainer with mixed precision and monitoring
    """
    def __init__(
        self,
        model: VoxeLinguaModel,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        config: TrainingConfig,
        device: torch.device
    ):
        self.config = config
        self.device = device
        
        # Initialize distributed training
        if config.distributed:
            self.setup_distributed()
            self.model = DDP(
                model.to(device),
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
        else:
            self.model = model.to(device)
            
        # Data loaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Optimizer and scheduler
        self.optimizer = AdaptiveOptimizer(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        
        # Training state
        self.state = TrainingState()
        
        # Monitoring
        if self.is_main_process:
            self.setup_monitoring()
            
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        
    def setup_distributed(self):
        """Initialize distributed training environment"""
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.local_rank)
        
    def setup_monitoring(self):
        """Initialize wandb and other monitoring tools"""
        wandb.init(
            project=self.config.project_name,
            name=self.config.run_name,
            config=self.config.__dict__
        )
        
    @property
    def is_main_process(self) -> bool:
        """Check if current process is the main process"""
        if self.config.distributed:
            return dist.get_rank() == 0
        return True
        
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save training checkpoint"""
        if not self.is_main_process:
            return
            
        self.state.model_state = self.model.state_dict()
        self.state.optimizer_state = self.optimizer.state_dict()
        self.state.scheduler_state = self.scheduler.state_dict()
        self.state.scaler_state = self.scaler.state_dict()
        
        self.state.save(path)
        
        if is_best:
            best_path = path.parent / 'best_model.pt'
            torch.save(self.model.state_dict(), best_path)
            
    def load_checkpoint(self, path: Path):
        """Load training checkpoint"""
        self.state = TrainingState.load(path)
        
        self.model.load_state_dict(self.state.model_state)
        self.optimizer.load_state_dict(self.state.optimizer_state)
        self.scheduler.load_state_dict(self.state.scheduler_state)
        self.scaler.load_state_dict(self.state.scaler_state)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with mixed precision"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with autocast():
            outputs = self.model(**batch)
            loss = self.criterion(
                outputs['logits'].view(-1, self.config.vocab_size),
                batch['labels'].view(-1)
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        if (self.state.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        # Calculate metrics
        metrics = {
            'loss': loss.item() * self.gradient_accumulation_steps,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'grad_norm': self.get_grad_norm()
        }
        
        return metrics
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        for batch in tqdm(self.val_dataloader, desc='Validation'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            loss = self.criterion(
                outputs['logits'].view(-1, self.config.vocab_size),
                batch['labels'].view(-1)
            )
            
            total_loss += loss.item() * batch['labels'].size(0)
            total_samples += batch['labels'].size(0)
            
        # Gather metrics from all processes
        if self.config.distributed:
            metrics = torch.tensor([total_loss, total_samples], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, total_samples = metrics.tolist()
            
        return {
            'val_loss': total_loss / total_samples
        }
        
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            
            # Training epoch
            if self.config.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
                
            epoch_metrics = []
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc=f'Epoch {epoch}')):
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                self.state.global_step += 1
                
                # Logging
                if self.is_main_process and self.state.global_step % self.config.logging_steps == 0:
                    avg_metrics = {
                        k: np.mean([m[k] for m in epoch_metrics[-self.config.logging_steps:]])
                        for k in metrics.keys()
                    }
                    wandb.log(avg_metrics, step=self.state.global_step)
                    
            # Validation
            val_metrics = self.validate()
            
            if self.is_main_process:
                wandb.log(val_metrics, step=self.state.global_step)
                
                # Save checkpoint
                self.save_checkpoint(
                    Path(self.config.output_dir) / f'checkpoint-{epoch}.pt',
                    is_best=val_metrics['val_loss'] < best_val_loss
                )
                
                best_val_loss = min(best_val_loss, val_metrics['val_loss'])
                
    def get_grad_norm(self) -> float:
        """Calculate gradient norm for monitoring"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
