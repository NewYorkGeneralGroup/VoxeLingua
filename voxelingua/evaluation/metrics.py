from typing import Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from dataclasses import dataclass
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    loss: float
    perplexity: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    rouge_scores: Dict[str, float]
    bleu_scores: Dict[str, float]
    attention_entropy: float
    confidence_scores: Dict[str, float]
    semantic_similarity: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            'loss': self.loss,
            'perplexity': self.perplexity,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            **{f'rouge_{k}': v for k, v in self.rouge_scores.items()},
            **{f'bleu_{k}': v for k, v in self.bleu_scores.items()},
            'attention_entropy': self.attention_entropy,
            **{f'confidence_{k}': v for k, v in self.confidence_scores.items()},
            'semantic_similarity': self.semantic_similarity
        }

class MetricsCalculator:
    """Advanced metrics calculation with sophisticated analysis"""
    def __init__(self, config):
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset accumulated metrics"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.predictions = []
        self.labels = []
        self.attention_weights = []
        self.confidence_scores = defaultdict(list)
        self.semantic_similarities = []
        
    def update(self, 
               outputs: Dict[str, torch.Tensor],
               labels: torch.Tensor,
               loss: float):
        """Update metrics with batch results"""
        batch_size = labels.size(0)
        self.total_samples += batch_size
        self.total_loss += loss * batch_size
        
        # Get predictions
        predictions = torch.argmax(outputs['logits'], dim=-1)
        
        # Store for later calculation
        self.predictions.extend(predictions.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        
        # Store attention weights
        if 'attention_weights' in outputs:
            self.attention_weights.append(
                outputs['attention_weights'].detach().cpu().numpy()
            )
            
        # Calculate confidence scores
        probs = torch.softmax(outputs['logits'], dim=-1)
        self.confidence_scores['max_prob'].extend(
            probs.max(dim=-1)[0].cpu().numpy()
        )
        self.confidence_scores['entropy'].extend(
            self._calculate_entropy(probs).cpu().numpy()
        )
        
        # Calculate semantic similarity if available
        if 'embeddings' in outputs:
            self.semantic_similarities.extend(
                self._calculate_semantic_similarity(
                    outputs['embeddings']
                ).cpu().numpy()
            )
            
    def _calculate_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate prediction entropy"""
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
    def _calculate_semantic_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate semantic similarity using embeddings"""
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        similarities = torch.matmul(
            normalized_embeddings,
            normalized_embeddings.transpose(-2, -1)
        )
        return similarities.mean(dim=-1)
        
    def _calculate_attention_entropy(self) -> float:
        """Calculate attention distribution entropy"""
        if not self.attention_weights:
            return 0.0
            
        attention_weights = np.concatenate(self.attention_weights, axis=0)
        entropy = -np.sum(
            attention_weights * np.log(attention_weights + 1e-10),
            axis=-1
        )
        return float(entropy.mean())
        
    def compute(self) -> EvaluationMetrics:
        """Compute final metrics"""
        # Basic metrics
        avg_loss = self.total_loss / self.total_samples
        perplexity = math.exp(avg_loss)
        
        # Classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.labels,
            self.predictions,
            average='weighted'
        )
        
        accuracy = np.mean(
            np.array(self.predictions) == np.array(self.labels)
        )
        
        # ROUGE scores
        rouge_scores = self._calculate_rouge_scores()
        
        # BLEU scores
        bleu_scores = self._calculate_bleu_scores()
        
        # Attention analysis
        attention_entropy = self._calculate_attention_entropy()
        
        # Confidence analysis
        confidence_scores = {
            k: float(np.mean(v))
            for k, v in self.confidence_scores.items()
        }
        
        # Semantic similarity
        semantic_similarity = float(np.mean(self.semantic_similarities)) if self.semantic_similarities else 0.0
        
        return EvaluationMetrics(
            loss=avg_loss,
            perplexity=perplexity,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            rouge_scores=rouge_scores,
            bleu_scores=bleu_scores,
            attention_entropy=attention_entropy,
            confidence_scores=confidence_scores,
            semantic_similarity=semantic_similarity
        )
        
    def _calculate_rouge_scores(self) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
            
            scores = defaultdict(list)
            for pred, label in zip(self.predictions, self.labels):
                score = scorer.score(pred, label)
                for k, v in score.items():
                    scores[k].append(v.fmeasure)
                    
            return {k: float(np.mean(v)) for k, v in scores.items()}
        except ImportError:
            logger.warning("rouge_score not installed. Skipping ROUGE calculation.")
            return {}
            
    def _calculate_bleu_scores(self) -> Dict[str, float]:
        """Calculate BLEU scores"""
        try:
            from sacrebleu.metrics import BLEU
            bleu = BLEU()
            
            scores = {}
            for n in [1, 2, 3, 4]:
                score = bleu.corpus_score(
                    self.predictions,
                    [self.labels],
                    force=True,
                    tokenize='none'
                )
                scores[f'bleu{n}'] = float(score.score)
                
            return scores
        except ImportError:
            logger.warning("sacrebleu not installed. Skipping BLEU calculation.")
            return {}
