"""
Dataset implementation for Deep Tree Echo training
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, Optional


class EchoDataset(Dataset):
    """
    Dataset for Deep Tree Echo LLM training.
    Handles tokenized sequences with proper windowing and padding.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        max_length: int = 2048,
        stride: int = 512,
        split: str = 'train'
    ):
        self.data = data
        self.max_length = max_length
        self.stride = stride
        self.split = split
        
        # Calculate number of samples
        self.n_samples = max(1, (len(data) - max_length) // stride + 1)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing input_ids and labels
        """
        # Calculate start and end positions
        start = idx * self.stride
        end = min(start + self.max_length, len(self.data))
        
        # Handle edge case where we don't have enough data
        if end - start < self.max_length:
            # Pad with zeros or wrap around
            sequence = np.zeros(self.max_length, dtype=np.int64)
            actual_length = end - start
            sequence[:actual_length] = self.data[start:end]
        else:
            sequence = self.data[start:end].astype(np.int64)
        
        # Convert to tensor
        input_ids = torch.from_numpy(sequence[:-1])
        labels = torch.from_numpy(sequence[1:])
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != 0).float()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    def get_batch(self, indices: list) -> Dict[str, torch.Tensor]:
        """
        Get a batch of samples.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Batched dictionary of tensors
        """
        batch = [self[idx] for idx in indices]
        
        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }