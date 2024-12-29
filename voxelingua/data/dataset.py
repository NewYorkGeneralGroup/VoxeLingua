from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import random
from concurrent.futures import ThreadPoolExecutor
import mmap
import json

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for dataset processing"""
    max_sequence_length: int = 512
    vocab_size: int = 50000
    mask_probability: float = 0.15
    random_seed: int = 42
    num_workers: int = 4
    buffer_size: int = 10000
    preprocessing_batch_size: int = 1000
    cache_dir: str = "cache"
    use_memory_mapping: bool = True
    dynamic_masking: bool = True
    whole_word_masking: bool = True
    span_masking: bool = True
    max_span_length: int = 10

class VoxeLinguaDataset(Dataset):
    """
    Advanced dataset implementation with sophisticated preprocessing
    """
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        is_training: bool = True
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training
        
        # Initialize random state
        self.rng = np.random.RandomState(config.random_seed)
        
        # Setup memory mapping for efficient data loading
        if config.use_memory_mapping:
            self._setup_memory_mapping()
        else:
            self._load_data()
            
        # Initialize caching system
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup preprocessing workers
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        
    def _setup_memory_mapping(self):
        """Setup memory mapping for efficient data access"""
        self.data_file = open(self.data_path, 'rb')
        self.mm = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Load index information
        index_path = self.data_path.with_suffix('.index')
        with open(index_path, 'r') as f:
            self.index = json.load(f)
            
        self.length = len(self.index)
        
    def _load_data(self):
        """Load entire dataset into memory"""
        with h5py.File(self.data_path, 'r') as f:
            self.data = {
                'input_ids': f['input_ids'][:],
                'attention_mask': f['attention_mask'][:]
            }
        self.length = len(self.data['input_ids'])
        
    def _get_item_from_mmap(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieve item using memory mapping"""
        start, end = self.index[idx]['span']
        self.mm.seek(start)
        data = self.mm.read(end - start)
        return json.loads(data)
        
    def _apply_masking(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        """Apply sophisticated masking strategies"""
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)
        
        if self.config.whole_word_masking:
            # Group tokens into words using tokenizer
            word_starts = []
            for i, token in enumerate(tokens):
                if not token.startswith('##'):
                    word_starts.append(i)
                    
            # Mask whole words
            for start in word_starts:
                if self.rng.random() < self.config.mask_probability:
                    end = start + 1
                    while end < len(tokens) and tokens[end].startswith('##'):
                        end += 1
                        
                    # Apply different masking strategies
                    if self.rng.random() < 0.8:
                        # Mask tokens
                        for i in range(start, end):
                            masked_tokens[i] = self.tokenizer.mask_token_id
                            labels[i] = tokens[i]
                    elif self.rng.random() < 0.5:
                        # Replace with random tokens
                        for i in range(start, end):
                            masked_tokens[i] = self.rng.randint(0, self.config.vocab_size)
                            labels[i] = tokens[i]
                            
        elif self.config.span_masking:
            # Implement span masking
            i = 0
            while i < len(tokens):
                if self.rng.random() < self.config.mask_probability:
                    span_length = min(
                        self.rng.randint(1, self.config.max_span_length + 1),
                        len(tokens) - i
                    )
                    
                    # Apply span masking
                    for j in range(i, i + span_length):
                        if self.rng.random() < 0.8:
                            masked_tokens[j] = self.tokenizer.mask_token_id
                        elif self.rng.random() < 0.5:
                            masked_tokens[j] = self.rng.randint(0, self.config.vocab_size)
                        labels[j] = tokens[j]
                        
                    i += span_length
                else:
                    i += 1
                    
        else:
            # Standard token masking
            for i, token in enumerate(tokens):
                if self.rng.random() < self.config.mask_probability:
                    labels[i] = token
                    if self.rng.random() < 0.8:
                        masked_tokens[i] = self.tokenizer.mask_token_id
                    elif self.rng.random() < 0.5:
                        masked_tokens[i] = self.rng.randint(0, self.config.vocab_size)
                        
        return masked_tokens, labels
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get preprocessed item with sophisticated masking"""
        if self.config.use_memory_mapping:
            item = self._get_item_from_mmap(idx)
        else:
            item = {
                'input_ids': self.data['input_ids'][idx],
                'attention_mask': self.data['attention_mask'][idx]
            }
            
        # Apply dynamic masking during training
        if self.is_training and self.config.dynamic_masking:
            masked_inputs, labels = self._apply_masking(item['input_ids'])
            item['input_ids'] = masked_inputs
            item['labels'] = labels
            
        # Convert to tensors
        return {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in item.items()
        }
        
    def __len__(self) -> int:
        return self.length
        
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'data_file'):
            self.data_file.close()
        self.executor.shutdown()

class DataCollator:
    """
    Advanced data collation with dynamic batching
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig
    ):
        self.tokenizer = tokenizer
        self.config = config
        
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate examples with dynamic padding"""
        # Determine batch size and sequence length
        batch_size = len(examples)
        max_length = max(len(ex['input_ids']) for ex in examples)
        
        # Initialize tensors
        input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.ones((batch_size, max_length), dtype=torch.long) * -100
        
        # Fill tensors
        for i, example in enumerate(examples):
            input_length = len(example['input_ids'])
            input_ids[i, :input_length] = example['input_ids']
            attention_mask[i, :input_length] = example['attention_mask']
            if 'labels' in example:
                labels[i, :input_length] = example['labels']
                
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def create_dataloader(
    dataset: VoxeLinguaDataset,
    batch_size: int,
    collator: DataCollator,
    is_training: bool = True
) -> DataLoader:
    """Create dataloader with advanced features"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=dataset.config.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=is_training
    )
