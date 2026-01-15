"""
Data loader for Deep Tree Echo training
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path

from .dataset import EchoDataset
from .preprocessor import DataPreprocessor


class EchoDataLoader:
    """
    Data loader for Deep Tree Echo LLM training.
    Handles data preparation, preprocessing, and batching.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data/deep_echo'))
        self.dataset_name = config.get('dataset_name', 'deep_echo_corpus')
        
        # Preprocessing config
        self.preprocessing = config.get('preprocessing', {})
        self.tokenizer_name = self.preprocessing.get('tokenizer', 'gpt2')
        self.max_length = self.preprocessing.get('max_length', 2048)
        self.stride = self.preprocessing.get('stride', 512)
        
        # Augmentation config
        self.augmentation = config.get('augmentation', {})
        
        # Loader config
        self.loader_config = config.get('loader', {})
        self.batch_size = self.loader_config.get('batch_size', 8)
        self.num_workers = self.loader_config.get('num_workers', 4)
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(self.preprocessing, self.augmentation)
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Check for existing processed data
        train_path = self.data_dir / 'train.bin'
        val_path = self.data_dir / 'val.bin'
        
        if train_path.exists() and val_path.exists():
            print(f"Loading existing processed data from {self.data_dir}")
            train_dataset = self._load_processed_dataset(train_path, 'train')
            val_dataset = self._load_processed_dataset(val_path, 'val')
        else:
            print(f"Generating synthetic data for {self.dataset_name}")
            train_dataset, val_dataset = self._generate_synthetic_data()
        
        return train_dataset, val_dataset
    
    def _load_processed_dataset(self, path: Path, split: str) -> EchoDataset:
        """Load a processed dataset from disk."""
        data = np.memmap(path, dtype=np.uint16, mode='r')
        
        # Convert to proper dataset
        dataset = EchoDataset(
            data=data,
            max_length=self.max_length,
            stride=self.stride,
            split=split
        )
        
        return dataset
    
    def _generate_synthetic_data(self) -> Tuple[EchoDataset, EchoDataset]:
        """
        Generate synthetic training data for Deep Tree Echo model.
        This creates data with hierarchical patterns suitable for tree attention.
        """
        print("Generating synthetic Deep Tree Echo training data...")
        
        # Generate text with hierarchical structure
        train_texts = self._generate_hierarchical_texts(n_samples=1000, split='train')
        val_texts = self._generate_hierarchical_texts(n_samples=200, split='val')
        
        # Preprocess texts
        train_data = self.preprocessor.process_texts(train_texts)
        val_data = self.preprocessor.process_texts(val_texts)
        
        # Save processed data
        train_path = self.data_dir / 'train.bin'
        val_path = self.data_dir / 'val.bin'
        
        # Save as binary files
        train_array = np.array(train_data, dtype=np.uint16)
        val_array = np.array(val_data, dtype=np.uint16)
        
        train_array.tofile(train_path)
        val_array.tofile(val_path)
        
        print(f"Saved training data: {len(train_array)} tokens")
        print(f"Saved validation data: {len(val_array)} tokens")
        
        # Create datasets
        train_dataset = EchoDataset(
            data=train_array,
            max_length=self.max_length,
            stride=self.stride,
            split='train'
        )
        
        val_dataset = EchoDataset(
            data=val_array,
            max_length=self.max_length,
            stride=self.stride,
            split='val'
        )
        
        return train_dataset, val_dataset
    
    def _generate_hierarchical_texts(self, n_samples: int, split: str) -> list:
        """
        Generate texts with hierarchical and recursive patterns.
        """
        texts = []
        
        # Templates for hierarchical text generation
        templates = [
            "The deep tree echo architecture consists of {levels} levels. "
            "At level {level}, we have {branches} branches, each processing {concept}. "
            "The echo mechanism propagates information through {depth} recursive layers. "
            "This creates a hierarchical representation with {features} emergent features.",
            
            "In the recursive attention mechanism, the model performs {iterations} iterations. "
            "Each iteration deepens the understanding by {factor}%. "
            "The tree structure branches into {branches} paths, exploring {aspects} different aspects. "
            "Echo connections maintain {memory}% of previous state information.",
            
            "Hierarchical pooling operates across {scales} different scales. "
            "The finest scale captures {fine_detail}, while the coarsest captures {coarse_detail}. "
            "Between these scales, {intermediate} intermediate representations emerge. "
            "The echo layer maintains temporal coherence across {time_steps} time steps.",
            
            "The memory bank stores {capacity} key representations. "
            "Query operations retrieve relevant information with {accuracy}% accuracy. "
            "The tree attention mechanism explores {paths} different reasoning paths. "
            "Recursive processing depth reaches {max_depth} levels of abstraction.",
            
            "Deep learning in tree structures requires {components} key components: "
            "{component_1}, {component_2}, {component_3}, and {component_4}. "
            "Each component interacts through {interactions} types of connections. "
            "The echo effect amplifies important signals by {amplification}x.",
        ]
        
        # Concepts for filling templates
        concepts = {
            'levels': [3, 4, 5, 6, 7],
            'branches': [2, 4, 8, 16],
            'concept': ['tokens', 'embeddings', 'features', 'representations', 'patterns'],
            'depth': [2, 3, 4, 5],
            'features': [10, 20, 50, 100],
            'iterations': [3, 5, 7, 10],
            'factor': [10, 15, 20, 25],
            'aspects': [3, 5, 7, 9],
            'memory': [70, 80, 90, 95],
            'scales': [3, 4, 5],
            'fine_detail': ['individual tokens', 'character patterns', 'word boundaries'],
            'coarse_detail': ['document structure', 'semantic themes', 'discourse patterns'],
            'intermediate': [2, 3, 4, 5],
            'time_steps': [10, 20, 50, 100],
            'capacity': [128, 256, 512, 1024],
            'accuracy': [85, 90, 95, 98],
            'paths': [4, 8, 16, 32],
            'max_depth': [3, 5, 7, 10],
            'components': [4, 5, 6],
            'component_1': ['attention mechanism', 'tree structure', 'echo layers'],
            'component_2': ['recursive processing', 'hierarchical pooling', 'memory bank'],
            'component_3': ['gating functions', 'normalization layers', 'dropout'],
            'component_4': ['residual connections', 'positional encoding', 'output projection'],
            'interactions': [3, 5, 7],
            'amplification': [1.5, 2.0, 2.5, 3.0]
        }
        
        import random
        
        for i in range(n_samples):
            # Choose a random template
            template = random.choice(templates)
            
            # Fill in the template with random values
            text = template
            for key, values in concepts.items():
                if '{' + key + '}' in text:
                    value = random.choice(values)
                    text = text.replace('{' + key + '}', str(value))
            
            # Add some variation with prefixes and suffixes
            prefixes = [
                "Consider the following: ",
                "It is important to note that ",
                "Research has shown that ",
                "In practice, we observe that ",
                "The key insight is that ",
            ]
            
            suffixes = [
                " This demonstrates the power of hierarchical processing.",
                " The recursive nature enables complex reasoning.",
                " Echo connections provide temporal stability.",
                " Tree structures capture compositional semantics.",
                " This architecture scales efficiently with depth.",
            ]
            
            if random.random() > 0.5:
                text = random.choice(prefixes) + text
            if random.random() > 0.5:
                text = text + random.choice(suffixes)
            
            texts.append(text)
            
            # Add some longer, multi-paragraph texts
            if i % 10 == 0:
                paragraphs = [text]
                for _ in range(random.randint(2, 4)):
                    template = random.choice(templates)
                    para = template
                    for key, values in concepts.items():
                        if '{' + key + '}' in para:
                            value = random.choice(values)
                            para = para.replace('{' + key + '}', str(value))
                    paragraphs.append(para)
                texts.append('\n\n'.join(paragraphs))
        
        return texts
    
    def create_data_loaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training and validation.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.loader_config.get('pin_memory', True),
            persistent_workers=self.loader_config.get('persistent_workers', True),
            prefetch_factor=self.loader_config.get('prefetch_factor', 2)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.loader_config.get('pin_memory', True),
            persistent_workers=self.loader_config.get('persistent_workers', True),
            prefetch_factor=self.loader_config.get('prefetch_factor', 2)
        )
        
        return train_loader, val_loader