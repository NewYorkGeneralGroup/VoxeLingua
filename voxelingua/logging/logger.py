from typing import Any, Dict, Optional
import logging
import wandb
from pathlib import Path
import json
import time
from datetime import datetime
import torch.distributed as dist
import sys

class AdvancedLogger:
    """Sophisticated logging system with multiple backends"""
    def __init__(
        self,
        config,
        experiment_name: str,
        output_dir: Path,
        is_main_process: bool = True
    ):
        self.config = config
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.is_main_process = is_main_process
        
        # Setup directory structure
        self.setup_directories()
        
        # Initialize logging backends
        self.setup_logging()
        
        # Initialize metrics history
        self.metrics_history = defaultdict(list)
        
        # Setup timing
        self.step_times = []
        self.start_time = time.time()
        
    def setup_directories(self):
        """Create necessary directories"""
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'logs').mkdir(exist_ok=True)
            (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
            (self.output_dir / 'metrics').mkdir(exist_ok=True)
            
    def setup_logging(self):
        """Initialize logging backends"""
        if self.is_main_process:
            # File handler
            file_handler = logging.FileHandler(
                self.output_dir / 'logs' / f'{self.experiment_name}.log'
            )
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
            # Initialize wandb
            if self.config.use_wandb:
                wandb.init(
                    project=self.config.project_name,
                    name=self.experiment_name,
                    config=self.config.__dict__,
                    dir=str(self.output_dir)
                )
                
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ''
    ):
        """Log metrics with sophisticated handling"""
        if not self.is_main_process:
            return
            
        # Add timing information
        metrics['steps_per_second'] = self.calculate_throughput()
        metrics['time_elapsed'] = time.time() - self.start_time
        
        # Add prefix to metric names
        if prefix:
            metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}
            
        # Update history
        for k, v in metrics.items():
            self.metrics_history[k].append((step, v))
            
        # Log to wandb
        if self.config.use_wandb:
            wandb.log(metrics, step=step)
            
        # Log to file
        metrics_file = self.output_dir / 'metrics' / f'step_{step}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Log to console
        metrics_str = ' '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
        logging.info(f'Step {step}: {metrics_str}')
        
    def log_model_artifacts(
        self,
        model_outputs: Dict[str, Any],
        step: int
    ):
        """Log model-specific artifacts"""
        if not self.is_main_process:
            return
            
        artifacts_dir = self.output_dir / 'artifacts' / f'step_{step}'
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save attention visualizations
        if 'attention_weights' in model_outputs:
            self.save_attention_visualization(
                model_outputs['attention_weights'],
                artifacts_dir / 'attention.png'
            )
            
        # Save embedding projections
        if 'embeddings' in model_outputs:
            self.save_embedding_projection(
                model_outputs['embeddings'],
                artifacts_dir / 'embeddings.png'
            )
            
        # Log to wandb
        if self.config.use_wandb:
            wandb.log({
                'artifacts': wandb.Image(str(artifacts_dir))
            }, step=step)
            
    def calculate_throughput(self) -> float:
        """Calculate training throughput"""
        if len(self.step_times) < 2:
            return 0.0
            
        time_diffs = np.diff(self.step_times)
        return 1.0 / np.mean(time_diffs)
        
    def save_attention_visualization(
        self,
        attention_weights: torch.Tensor,
        output_path: Path
    ):
        """Generate attention visualization"""
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            attention_weights.mean(0).cpu().numpy(),
            cmap='viridis'
        )
        plt.savefig(output_path)
        plt.close()
        
    def save_embedding_projection(
        self,
        embeddings: torch.Tensor,
        output_path: Path
    ):
        """Generate embedding projection visualization"""
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Project to 2D
        tsne = TSNE(n_components=2, random_state=42)
        projections = tsne.fit_transform(embeddings.cpu().numpy())
        
        plt.figure(figsize=(10, 10))
        plt.scatter(projections[:, 0], projections[:, 1], alpha=0.5)
        plt.savefig(output_path)
        plt.close()
        
    def close(self):
        """Cleanup logging resources"""
        if self.is_main_process and self.config.use_wandb:
            wandb.finish()
