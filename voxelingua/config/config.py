from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
import json
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    feature_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    max_sequence_length: int = 512
    vocab_size: int = 50000
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    use_gradient_checkpointing: bool = False
    use_memory_efficient_attention: bool = True
    attention_window_size: int = 512
    position_embedding_type: str = "rotary"
    activation_function: str = "gelu_new"
    tie_word_embeddings: bool = True
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "outputs"
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    fp16_opt_level: str = "O2"
    local_rank: int = -1
    seed: int = 42
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 5
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    label_smoothing: float = 0.1
    
@dataclass
class Config:
    """Master configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        return cls(model=model_config, training=training_config, data=data_config)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save(self, path: Path):
        """Save config to file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f)
            
    def update(self, **kwargs):
        """Update config parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
                
    def validate(self):
        """Validate configuration parameters"""
        # Validate model config
        assert self.model.feature_dim % self.model.num_heads == 0, \
            "Feature dimension must be divisible by number of heads"
        assert self.model.max_sequence_length <= 512, \
            "Maximum sequence length exceeds model capacity"
            
        # Validate training config
        if self.training.fp16:
            assert torch.cuda.is_available(), \
                "FP16 training requires CUDA"
                
        # Validate data config
        assert self.data.max_sequence_length <= self.model.max_sequence_length, \
            "Data sequence length exceeds model capacity"
